CREATE TABLE `esi_sessions` (
  `id` text PRIMARY KEY NOT NULL,
  `expires` integer NOT NULL
);
--> statement-breakpoint
CREATE TABLE `esi_drafts` (
  `id` text PRIMARY KEY NOT NULL,
  `owner` text NOT NULL,
  `expires` integer NOT NULL,
  `epoch` integer NOT NULL DEFAULT 0,
  `data` text NOT NULL
);
--> statement-breakpoint
CREATE INDEX `esi_drafts_owner` ON `esi_drafts` (`owner`);
--> statement-breakpoint
CREATE INDEX `esi_drafts_expiry` ON `esi_drafts` (`expires`);
--> statement-breakpoint
CREATE TABLE `esi_limits` (
  `id` text PRIMARY KEY NOT NULL,
  `count` integer NOT NULL,
  `expires` integer NOT NULL
);
