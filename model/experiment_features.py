"""Fixed group routing and reference-only target-neighbor features."""
from pathlib import Path
import json
import numpy as np
from scipy.spatial import cKDTree
from sklearn.ensemble import HistGradientBoostingClassifier
from baseline import LABELS, MODEL_PARAMETERS, NUMERIC_FEATURES, SOURCE_VITALS, make_pipeline

AGE_BOUNDS = [40, 65]
MIN_GROUP_TRAIN = 500
NEIGHBOR_K = 20
NEIGHBOR_NAMES = ['nearest_esi', 'neighbor_esi_mean', 'neighbor_esi_std',
                  'neighbor_count', 'exact_complaint_pool']
# Fixed unsupervised taxonomy. This is a routing experiment, not ESI rules.
FAMILY_MEMBERS = {
    'cardiac_chest': 'cardiacarrest chestpain chesttightness irregularheartbeat palpitations rapidheartrate tachycardia hypertension hypotension',
    'respiratory': 'asthma breathingdifficulty breathingproblem cough dyspnea hemoptysis respiratorydistress shortnessofbreath wheezing',
    'abdominal_gi': 'abdominalcramping abdominaldistention abdominalpain constipation diarrhea emesis epigastricpain gibleeding giproblem nausea rectalbleeding rectalpain swallowedforeignbody',
    'neurologic': 'alteredmentalstatus confusion dizziness extremityweakness headache headache-newonsetornewsymptoms headache-recurrentorknowndxmigraines headachere-evaluation lethargy lossofconsciousness migraine nearsyncope neurologicproblem numbness seizure-newonset seizure-priorhxof seizures strokealert syncope unresponsive',
    'injury_trauma': 'animalbite arminjury assaultvictim ankleinjury burn elbowpain eyeinjury facialinjury faciallaceration fall fall>65 fingerinjury footinjury fulltrauma handinjury headinjury headlaceration kneeinjury laceration leginjury modifiedtrauma motorcyclecrash motorvehiclecrash multiplefalls ribinjury shoulderinjury thumbinjury toeinjury trauma wristinjury extremitylaceration',
    'musculoskeletal': 'anklepain armpain armswelling backpain fingerpain fingerswelling footpain footswelling generalizedbodyaches handpain hippain jawpain jointswelling kneepain legpain legswelling neckpain ribpain shoulderpain toepain wristpain',
    'mental_health_substance': 'addictionproblem agitation alcoholintoxication alcoholproblem anxiety depression detoxevaluation drug/alcoholassessment drugproblem hallucinations homicidal ingestion overdose-accidental overdose-intentional panicattack poisoning psychiatricevaluation psychoticsymptoms suicidal withdrawal-alcohol',
    'gu_reproductive': 'abdominalpainpregnant breastpain dysuria exposuretostd femaleguproblem flankpain groinpain hematuria maleguproblem pelvicpain stdcheck testiclepain urinaryfrequency urinaryretention urinarytractinfection vaginalbleeding vaginaldischarge vaginalpain',
    'skin_wound': 'abscess cellulitis cyst follow-upcellulitis insectbite mass rash skinirritation skinproblem suture/stapleremoval tickremoval woundcheck woundinfection woundre-evaluation',
    'eye_ent': 'blurredvision conjunctivitis dentalpain earpain earproblem eyepain eyeproblem eyeredness facialpain facialswelling foreignbodyineye nasalcongestion oralswelling otalgia sinusproblem sorethroat',
    'infectious_fever': 'chills coldlikesymptoms fever fever-75yearsorolder fever-9weeksto74years feverimmunocompromised influenza uri',
    'metabolic_hematologic': 'bleeding/bruising decreasedbloodsugar-symptomatic dehydration elevatedbloodsugar-nosymptoms elevatedbloodsugar-symptomatic hyperglycemia sicklecellpain',
}


def complaint_mapping(columns):
    mapping = {}
    for family, words in FAMILY_MEMBERS.items():
        for word in words.split():
            col = 'cc_' + word
            if col not in columns:
                raise ValueError(f'Unknown complaint in fixed taxonomy: {col}')
            if col in mapping:
                raise ValueError(f'Duplicate complaint assignment: {col}')
            mapping[col] = family
    return {c: mapping.get(c, 'other_general') for c in columns}


def complaint_signatures(x, columns):
    # Exact binary fingerprint includes missing-complaint status; never a target.
    present = x[columns].fillna(0).to_numpy() > 0
    binary = np.column_stack([present, x['complaint_missing'].to_numpy() > 0])
    packed = np.ascontiguousarray(np.packbits(binary, axis=1))
    return packed.view(f'V{packed.shape[1]}').ravel()


def routing_groups(x, columns):
    mapping = complaint_mapping(columns)
    families = sorted(set(mapping.values()))
    flags = np.column_stack([
        x[[c for c in columns if mapping[c] == f]].fillna(0).gt(0).any(axis=1).to_numpy()
        for f in families])
    counts = flags.sum(axis=1)
    complaint = np.array(['missing_complaint'] * len(x), dtype=object)
    one = counts == 1
    complaint[one] = np.array(families)[flags[one].argmax(axis=1)]
    complaint[counts > 1] = 'multiple_families'
    age_values = x['age'].to_numpy()
    age = np.select([np.isnan(age_values), age_values < 40, age_values < 65],
                    ['missing_age', '18_to_39', '40_to_64'], default='65_plus')
    arrival_values = x['ambulance'].to_numpy()
    arrival = np.select([np.isnan(arrival_values), arrival_values == 1],
                        ['unknown_arrival', 'ambulance'], default='non_ambulance')
    joint = np.array([f'{a}|{b}|{c}' for a, b, c in zip(complaint, age, arrival)], dtype=str)
    return {'none': np.array(['all'] * len(x)), 'complaint': complaint.astype(str),
            'age': age, 'arrival': arrival, 'joint': joint}


class TargetNeighbors:
    """Complaint-conditioned exact neighbors with reference-only preprocessing.

    Prefer the identical structured complaint fingerprint. If unseen, use the
    full reference pool and mark the fallback. Within a pool, find nearest
    standardized age/vitals with missingness and categorical distance terms.
    No query targets are accepted. For k statistics use up to 20 pool members.
    """
    def __init__(self, columns, k=NEIGHBOR_K):
        self.columns = list(columns)
        self.k = k

    def _encode(self, x):
        continuous = x[['age'] + SOURCE_VITALS[:-1] + ['temperature_c']].to_numpy(dtype=float)
        missing = np.isnan(continuous)
        values = np.where(missing, self.medians_, continuous)
        values = (values - self.centers_) / self.scales_
        gender = x['gender'].to_numpy()
        ambulance = x['ambulance'].to_numpy()
        return np.ascontiguousarray(np.column_stack([
            values, missing.astype(float) * .5,
            np.column_stack([gender == v for v in ['Female', 'Male', 'Unknown']]) * .5,
            np.column_stack([ambulance == 0, ambulance == 1, np.isnan(ambulance)]) * .5,
        ]), dtype=np.float64)

    def fit(self, x, reference_y, reference_ids):
        values = x[['age'] + SOURCE_VITALS[:-1] + ['temperature_c']].to_numpy(dtype=float)
        self.medians_ = np.nanmedian(values, axis=0)
        self.medians_[~np.isfinite(self.medians_)] = 0
        filled = np.where(np.isnan(values), self.medians_, values)
        self.centers_, self.scales_ = filled.mean(axis=0), filled.std(axis=0)
        self.scales_[self.scales_ == 0] = 1
        self.reference_y_ = np.asarray(reference_y, dtype=np.int8).copy()
        self.reference_ids_ = np.asarray(reference_ids, dtype=np.int64).copy()
        if len(np.unique(self.reference_ids_)) != len(x):
            raise ValueError('Reference row IDs must be unique.')
        signatures = complaint_signatures(x, self.columns)
        self.encoded_ = self._encode(x)
        self.global_tree_ = cKDTree(self.encoded_, leafsize=40)
        self.pools_ = {}
        for signature in np.unique(signatures):
            positions = np.flatnonzero(signatures == signature)
            self.pools_[bytes(signature)] = (positions, cKDTree(self.encoded_[positions], leafsize=40))
        return self

    def transform(self, query_x, *, query_ids=None, progress=None):
        if query_ids is not None and np.intersect1d(query_ids, self.reference_ids_).size:
            raise ValueError('A query row occurs in its target-neighbor reference pool.')
        encoded = self._encode(query_x)
        signatures = complaint_signatures(query_x, self.columns)
        output = np.empty((len(query_x), 5), dtype=np.float32)
        done = 0
        for signature in np.unique(signatures):
            query_positions = np.flatnonzero(signatures == signature)
            found = self.pools_.get(bytes(signature))
            exact = found is not None
            if exact:
                positions, tree = found
            else:
                positions, tree = None, self.global_tree_
            k = min(self.k, tree.n)
            # Chunk queries to bound memory for a common complaint pool.
            for start in range(0, len(query_positions), 4000):
                q = query_positions[start:start + 4000]
                _, nearest = tree.query(encoded[q], k=k, workers=4)
                nearest = np.asarray(nearest).reshape(len(q), k)
                index = nearest if positions is None else positions[nearest]
                labels = self.reference_y_[index].astype(float)
                output[q, 0] = labels[:, 0]
                output[q, 1] = labels.mean(axis=1)
                output[q, 2] = labels.std(axis=1, ddof=0)
                output[q, 3] = k
                output[q, 4] = float(exact)
            done += len(query_positions)
            if progress is not None and done >= getattr(self, '_next_progress', 50000):
                progress(done, len(query_x))
                self._next_progress = done + 50000
        assert np.isfinite(output).all()
        return output


def neighbor_columns(mode):
    return {'none': [], 'both': [0, 1, 2, 3, 4],
            'nearest': [0, 4], 'stats': [1, 2, 3, 4]}[mode]


def training_weights(y, weighted):
    # Define weights on recorded ESI before the classifier encodes class labels.
    return np.where(np.asarray(y) == 2, 2., 1.) if weighted else None


def fit_classifier(x, y, weighted):
    model = HistGradientBoostingClassifier(**MODEL_PARAMETERS)
    return model.fit(x, y, sample_weight=training_weights(y, weighted))


def aligned_probabilities(model, x):
    p = np.zeros((len(x), 5), dtype=np.float32)
    p[:, np.asarray(model.classes_, dtype=int) - 1] = model.predict_proba(x).astype(np.float32)
    return p


class ExperimentModel:
    """Research estimator with group fallback and optional target-neighbor index."""
    def __init__(self, preprocessing, global_model, group_models, group_kind,
                 columns, neighbor_mode, neighbor_index=None):
        self.preprocessing = preprocessing
        self.global_model = global_model
        self.group_models = group_models
        self.group_kind = group_kind
        self.columns = list(columns)
        self.neighbor_mode = neighbor_mode
        self.neighbor_index = neighbor_index
        self.classes_ = LABELS.copy()

    def predict_proba(self, x):
        missing = x[SOURCE_VITALS[:-1] + ['temperature_c']].isna().sum(axis=1)
        if (missing > 1).any():
            raise ValueError('This research model requires zero or one missing vital.')
        encoded = self.preprocessing.transform(x)
        cols = neighbor_columns(self.neighbor_mode)
        if cols:
            extra = self.neighbor_index.transform(x)[:, cols]
            encoded = np.column_stack([encoded, extra]).astype(np.float32)
        probabilities = aligned_probabilities(self.global_model, encoded)
        if self.group_kind != 'none':
            groups = routing_groups(x, self.columns)[self.group_kind]
            for group, model in self.group_models.items():
                mask = groups == group
                if mask.any():
                    probabilities[mask] = aligned_probabilities(model, encoded[mask])
        return probabilities
