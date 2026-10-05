CREATE TABLE samples (
	id INTEGER NOT NULL, 
	path VARCHAR NOT NULL, 
	rel_path VARCHAR, 
	filename VARCHAR NOT NULL, 
	file_size_bytes INTEGER, 
	file_hash VARCHAR, 
	modified_at INTEGER, 
	scanned_at DATETIME DEFAULT CURRENT_TIMESTAMP NOT NULL, 
	duration_s FLOAT, 
	sample_rate INTEGER, 
	channels INTEGER, 
	bit_depth INTEGER, 
	file_format VARCHAR, 
	acid_bpm FLOAT, 
	acid_beats INTEGER, 
	root_note INTEGER, 
	chunks_read INTEGER, 
	is_favorite BOOLEAN NOT NULL, 
	is_hidden BOOLEAN NOT NULL, 
	user_tags TEXT, 
	ableton_tags TEXT, 
	PRIMARY KEY (id)
);
CREATE TABLE device_profiles (
	id INTEGER NOT NULL, 
	device_id VARCHAR NOT NULL, 
	name VARCHAR NOT NULL, 
	yaml_path VARCHAR NOT NULL, 
	config_json TEXT NOT NULL, 
	last_used DATETIME, 
	PRIMARY KEY (id), 
	UNIQUE (device_id)
);
CREATE TABLE labels (
	sample_id INTEGER NOT NULL, 
	provider VARCHAR NOT NULL, 
	kind VARCHAR NOT NULL, 
	label VARCHAR NOT NULL, 
	rank INTEGER NOT NULL, 
	confidence FLOAT, 
	PRIMARY KEY (sample_id, provider, kind, label)
);
CREATE TABLE descriptors (
	sample_id INTEGER NOT NULL, 
	provider VARCHAR NOT NULL, 
	name VARCHAR NOT NULL, 
	value FLOAT, 
	PRIMARY KEY (sample_id, provider, name)
);
CREATE TABLE missing_files (
	sample_id INTEGER NOT NULL, 
	path VARCHAR NOT NULL, 
	since INTEGER, 
	PRIMARY KEY (sample_id)
);
CREATE TABLE walked_dirs (
	path VARCHAR NOT NULL, 
	mtime_ns BIGINT NOT NULL, 
	listing TEXT NOT NULL, 
	PRIMARY KEY (path)
);
CREATE TABLE sononym_meta (
	id INTEGER NOT NULL, 
	sample_id INTEGER NOT NULL, 
	sononym_asset_id INTEGER, 
	classes TEXT, 
	class_strengths TEXT, 
	categories TEXT, 
	category_strengths TEXT, 
	pitch_class VARCHAR, 
	base_note FLOAT, 
	base_note_confidence FLOAT, 
	peak_db FLOAT, 
	rms_db FLOAT, 
	crest_factor FLOAT, 
	bpm FLOAT, 
	bpm_confidence FLOAT, 
	brightness FLOAT, 
	noisiness FLOAT, 
	harmonicity FLOAT, 
	class_signature TEXT, 
	category_signature TEXT, 
	pitch_confidence FLOAT, 
	PRIMARY KEY (id), 
	UNIQUE (sample_id), 
	FOREIGN KEY(sample_id) REFERENCES samples (id)
);
CREATE TABLE sample_features (
	id INTEGER NOT NULL, 
	sample_id INTEGER NOT NULL, 
	computed_at DATETIME, 
	sub_weight FLOAT, 
	transient_score FLOAT, 
	loop_confidence FLOAT, 
	is_pitched INTEGER, 
	bpm_reliable INTEGER, 
	bpm_corrected FLOAT, 
	timbral_norm TEXT, 
	derived_computed_at DATETIME, 
	spectral_balance FLOAT, 
	pitch_stability INTEGER, 
	attack_class VARCHAR, 
	drum_subtype VARCHAR, 
	spectral_centroid_mean FLOAT, 
	spectral_bandwidth_mean FLOAT, 
	spectral_rolloff_mean FLOAT, 
	spectral_flatness_mean FLOAT, 
	zero_crossing_rate_mean FLOAT, 
	rms_mean FLOAT, 
	tempo_bpm FLOAT, 
	onset_rate_hz FLOAT, 
	mfcc_mean TEXT, 
	attack_time_ms FLOAT, 
	decay_time_ms FLOAT, 
	harmonic_percussive_ratio FLOAT, 
	chroma_concentration FLOAT, 
	is_clipped INTEGER, 
	dc_offset_ratio FLOAT, 
	clap_embedding BLOB, 
	clap_model VARCHAR, 
	detected_key VARCHAR, 
	key_confidence FLOAT, 
	trim_end_s FLOAT, 
	trim_computed_at DATETIME, 
	n_events INTEGER, 
	event_regularity FLOAT, 
	event_echo INTEGER, 
	events_computed_at DATETIME, 
	PRIMARY KEY (id), 
	UNIQUE (sample_id), 
	FOREIGN KEY(sample_id) REFERENCES samples (id)
);
CREATE TABLE packs (
	id INTEGER NOT NULL, 
	name VARCHAR NOT NULL, 
	device_id INTEGER, 
	description TEXT, 
	created_at DATETIME DEFAULT CURRENT_TIMESTAMP NOT NULL, 
	export_path VARCHAR, 
	skill_params TEXT, 
	PRIMARY KEY (id), 
	FOREIGN KEY(device_id) REFERENCES device_profiles (id)
);
CREATE TABLE pack_items (
	id INTEGER NOT NULL, 
	pack_id INTEGER NOT NULL, 
	sample_id INTEGER NOT NULL, 
	dest_folder VARCHAR, 
	dest_filename VARCHAR, 
	slot_name VARCHAR, 
	cv_role VARCHAR, 
	sort_order INTEGER NOT NULL, 
	PRIMARY KEY (id), 
	CONSTRAINT uq_pack_sample UNIQUE (pack_id, sample_id), 
	FOREIGN KEY(pack_id) REFERENCES packs (id), 
	FOREIGN KEY(sample_id) REFERENCES samples (id)
);
CREATE TABLE schema_version (version INTEGER NOT NULL, applied_at TEXT NOT NULL);
CREATE INDEX ix_samples_filename ON samples (filename);
CREATE INDEX idx_samples_format ON samples (file_format);
CREATE INDEX ix_samples_file_hash ON samples (file_hash);
CREATE INDEX ix_samples_rel_path ON samples (rel_path);
CREATE UNIQUE INDEX ix_samples_path ON samples (path);
CREATE INDEX ix_labels_provider_kind_label ON labels (provider, kind, label);
CREATE INDEX ix_descriptors_provider_name ON descriptors (provider, name);
CREATE INDEX ix_sononym_meta_sononym_asset_id ON sononym_meta (sononym_asset_id);
CREATE INDEX ix_sononym_meta_bpm ON sononym_meta (bpm);
