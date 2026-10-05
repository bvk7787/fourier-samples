"""SQLAlchemy models for fourier."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from sqlalchemy import (
    BigInteger,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    LargeBinary,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy.types import TypeDecorator


class JSONText(TypeDecorator):
    """
    JSON stored as plain TEXT for cross-DB compatibility (SQLite and DuckDB).

    Storing as TEXT ensures LIKE queries work on both backends.
    Python API is identical to JSON: accepts and returns Python lists/dicts.
    """
    impl = Text
    cache_ok = True

    def process_bind_param(self, value, dialect):
        if value is not None:
            return json.dumps(value)
        return None

    def process_result_value(self, value, dialect):
        if value is not None:
            try:
                return json.loads(value)
            except (TypeError, ValueError):
                return value
        return None


class Base(DeclarativeBase):
    pass


class Sample(Base):
    """Core sample record - one row per audio file."""

    __tablename__ = "samples"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    # Absolute path on disk
    path: Mapped[str] = mapped_column(String, unique=True, nullable=False, index=True)
    # Path relative to library root (matches Sononym's filename column)
    rel_path: Mapped[str | None] = mapped_column(String, index=True)
    filename: Mapped[str] = mapped_column(String, nullable=False, index=True)

    # File-level metadata
    file_size_bytes: Mapped[int | None] = mapped_column(Integer)
    file_hash: Mapped[str | None] = mapped_column(String, index=True)  # sha256 first 8kb
    modified_at: Mapped[int | None] = mapped_column(Integer)  # unix timestamp
    scanned_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    # Audio format
    duration_s: Mapped[float | None] = mapped_column(Float)
    sample_rate: Mapped[int | None] = mapped_column(Integer)
    channels: Mapped[int | None] = mapped_column(Integer)
    bit_depth: Mapped[int | None] = mapped_column(Integer)
    file_format: Mapped[str | None] = mapped_column(String)  # wav, aif, mp3
    # What a WAV's own chunks state (ingest/chunks.py), read by the library walk: the ACID
    # chunk's tempo and beats, the root note (ACID, else the smpl chunk's unity note), and
    # whether the chunks were read (1), so a later walk reads a file only once
    acid_bpm: Mapped[float | None] = mapped_column(Float)
    acid_beats: Mapped[int | None] = mapped_column(Integer)
    root_note: Mapped[int | None] = mapped_column(Integer)
    chunks_read: Mapped[int | None] = mapped_column(Integer)

    # User metadata
    is_favorite: Mapped[bool] = mapped_column(Boolean, default=False)
    is_hidden: Mapped[bool] = mapped_column(Boolean, default=False)
    user_tags: Mapped[list | None] = mapped_column(JSONText)  # ["punchy", "techno", ...]
    ableton_tags: Mapped[list | None] = mapped_column(JSONText)  # Ableton Live auto-tags

    # Relationships
    sononym: Mapped[SononymMeta | None] = relationship(
        back_populates="sample", uselist=False, cascade="all, delete-orphan"
    )
    features: Mapped[SampleFeatures | None] = relationship(
        back_populates="sample", uselist=False, cascade="all, delete-orphan"
    )
    pack_items: Mapped[list[PackItem]] = relationship(back_populates="sample")

    __table_args__ = (Index("idx_samples_format", "file_format"),)

    def __repr__(self) -> str:
        return f"<Sample {self.filename}>"

    @property
    def mono(self) -> bool:
        return self.channels == 1

    @property
    def primary_category(self) -> str | None:
        """Best single category label, preferring Sononym data."""
        if self.sononym and self.sononym.categories:
            cats = self.sononym.categories
            if cats:
                return cats[0]
        return None

    @property
    def is_oneshot(self) -> bool | None:
        if self.sononym:
            classes = self.sononym.classes or []
            return "OneShot" in classes
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "path": self.path,
            "filename": self.filename,
            "duration_s": self.duration_s,
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "bit_depth": self.bit_depth,
            "file_format": self.file_format,
            "file_size_bytes": self.file_size_bytes,
            "is_favorite": self.is_favorite,
            "category": self.primary_category,
            "is_oneshot": self.is_oneshot,
            "sononym": self.sononym.to_dict() if self.sononym else None,
            "features": self.features.to_dict() if self.features else None,
        }


class SononymMeta(Base):
    """Metadata imported directly from Sononym's SQLite database."""

    __tablename__ = "sononym_meta"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    sample_id: Mapped[int] = mapped_column(ForeignKey("samples.id"), unique=True, nullable=False)
    sononym_asset_id: Mapped[int | None] = mapped_column(Integer, index=True)

    # Sononym classification
    classes: Mapped[list | None] = mapped_column(JSONText)        # ["OneShot"] or ["Loop"]
    class_strengths: Mapped[list | None] = mapped_column(JSONText)  # probability vector
    categories: Mapped[list | None] = mapped_column(JSONText)     # ["Perc Kicks"]
    category_strengths: Mapped[list | None] = mapped_column(JSONText)

    # Tonal analysis
    pitch_class: Mapped[str | None] = mapped_column(String)   # "C", "F#", etc.
    base_note: Mapped[float | None] = mapped_column(Float)    # MIDI note number
    base_note_confidence: Mapped[float | None] = mapped_column(Float)

    # Dynamics
    peak_db: Mapped[float | None] = mapped_column(Float)
    rms_db: Mapped[float | None] = mapped_column(Float)
    crest_factor: Mapped[float | None] = mapped_column(Float)

    # Rhythm
    bpm: Mapped[float | None] = mapped_column(Float, index=True)
    bpm_confidence: Mapped[float | None] = mapped_column(Float)

    # Timbral descriptors (Sononym's 0-1 normalized)
    brightness: Mapped[float | None] = mapped_column(Float)
    noisiness: Mapped[float | None] = mapped_column(Float)
    harmonicity: Mapped[float | None] = mapped_column(Float)

    # Extended signatures (stored as JSON arrays)
    class_signature: Mapped[list | None] = mapped_column(JSONText)     # [loop_prob, oneshot_prob]
    category_signature: Mapped[list | None] = mapped_column(JSONText)  # 29-dim category vector
    # spectrum_signature is large (~VVR blob) - skip for now, query from Sononym directly if needed
    pitch_confidence: Mapped[float | None] = mapped_column(Float)

    sample: Mapped[Sample] = relationship(back_populates="sononym")

    def to_dict(self) -> dict[str, Any]:
        return {
            "classes": self.classes,
            "categories": self.categories,
            "pitch_class": self.pitch_class,
            "base_note": self.base_note,
            "bpm": self.bpm,
            "bpm_confidence": self.bpm_confidence,
            "brightness": self.brightness,
            "noisiness": self.noisiness,
            "harmonicity": self.harmonicity,
            "peak_db": self.peak_db,
            "rms_db": self.rms_db,
        }


class SampleFeatures(Base):
    """
    Computed sample features. Every column holds values from one source only:

    - Tier 1 (derived, LEGACY, SONONYM-DERIVED): arithmetic on a sample's Sononym row, no
      file I/O (`fourier tools analyze --only derived`, tracked by `derived_computed_at`).
      Written only from a Sononym row: a sample without one gets NULL in every Tier 1 column
      (before schema 2 it got a stand-in 0 in bpm_reliable, which reads the same: not
      reliable). The names predate the provenance split and stay so a database keeps working
      as it is; read them as Sononym's.
    - Fourier's own measurements (Tier 2 librosa, events, quality, CLAP, key, loop trim, and
      the pYIN root `own_root_midi`): computed from the audio by Fourier, whether or not
      Sononym is there. `computed_at` is when librosa ran.
    - `sononym_bpm_folded`: Sononym's tempo folded into 60-200 (`--only bpm-fix`).
    - `bpm_corrected`: LEGACY, no longer written or read. On a database analysed before
      schema 2 it holds Sononym's folded tempo, or librosa's where Sononym had none.

    What the build uses, chosen from these and Sononym's own readings, and where each value
    came from, is metadata/resolve.py's.
    """

    __tablename__ = "sample_features"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    sample_id: Mapped[int] = mapped_column(ForeignKey("samples.id"), unique=True, nullable=False)
    computed_at: Mapped[datetime | None] = mapped_column(DateTime)  # when librosa features were run

    # ── Tier 1: LEGACY, Sononym-derived (from SononymMeta; NULL without a Sononym row) ─────

    # Sub-bass proxy: Sononym's harmonicity * (1 - brightness). ~0.4-0.7 for a good DnB sub kick.
    sub_weight: Mapped[float | None] = mapped_column(Float)

    # Punchiness proxy: Sononym's crest_factor / 8.0, clamped to 0-1. High = sharp transient.
    transient_score: Mapped[float | None] = mapped_column(Float)

    # Sononym's confidence that this is actually the class it was tagged as.
    # class_signature[0] if Loop, class_signature[1] if OneShot.
    loop_confidence: Mapped[float | None] = mapped_column(Float)

    # 1 if Sononym's harmonicity > 0.6 AND its pitch_confidence > 0.5 (Sononym's pitched flag;
    # Fourier's own is metadata/resolve.own_pitched).
    is_pitched: Mapped[int | None] = mapped_column(Integer)

    # 1 if Sononym's bpm IS NOT NULL AND its bpm_confidence > 0.6: Sononym's tempo is trustworthy
    # (NULL without a Sononym row; Fourier's own is metadata/resolve.own_tempo_confidence).
    bpm_reliable: Mapped[int | None] = mapped_column(Integer)

    # LEGACY (no longer written or read): Sononym's bpm folded to [60, 200], or librosa's
    # tempo_bpm where Sononym had none, as `--only bpm-fix` wrote it before schema 2. Its
    # successor for Sononym's tempo is sononym_bpm_folded; Fourier's own tempo is tempo_bpm.
    bpm_corrected: Mapped[float | None] = mapped_column(Float)

    # Sononym's bpm folded by octaves into [60, 200] (`fourier tools analyze --only bpm-fix`;
    # the migration to schema 2 sets it on an older database). NULL without a Sononym tempo.
    sononym_bpm_folded: Mapped[float | None] = mapped_column(Float)

    # Unit-L2-normalized Sononym [brightness, harmonicity, noisiness]: a timbral fingerprint.
    timbral_norm: Mapped[list | None] = mapped_column(JSONText)

    # When derived features were last computed (independent of librosa computed_at).
    derived_computed_at: Mapped[datetime | None] = mapped_column(DateTime)

    # ── Tier 1: later additions (LEGACY, Sononym-derived too) ──────────────────────────────

    # Low-to-high frequency energy ratio proxy: (1 - brightness) / (brightness + 0.05).
    # Higher = warmer/darker (kicks well above snares, snares above hats).
    spectral_balance: Mapped[float | None] = mapped_column(Float)

    # Stricter pitched-sample flag: 1 if harmonicity > 0.65 AND pitch_confidence > 0.65.
    # More conservative than is_pitched — signals samples safe to transpose chromatically.
    pitch_stability: Mapped[int | None] = mapped_column(Integer)

    # Human-readable attack descriptor from crest_factor + duration_s:
    # "snap" (very short + high crest), "punch" (medium), "swell" (slow/pad-like).
    attack_class: Mapped[str | None] = mapped_column(String)

    # Fine-grained within-category drum sub-classification.
    # e.g. "sub_kick" | "click_kick" | "acoustic_kick" | "snare" | "rimshot" | "clap"
    #    | "closed_hat" | "open_hat" | "cymbal" — None for non-drum categories.
    drum_subtype: Mapped[str | None] = mapped_column(String)

    # ── Tier 2: Fourier's own librosa features (reads audio files) ─────────────────────────

    # Spectral (lower priority — overlap with Sononym's brightness/noisiness)
    spectral_centroid_mean: Mapped[float | None] = mapped_column(Float)
    spectral_bandwidth_mean: Mapped[float | None] = mapped_column(Float)
    spectral_rolloff_mean: Mapped[float | None] = mapped_column(Float)
    spectral_flatness_mean: Mapped[float | None] = mapped_column(Float)  # 0=tonal, 1=white noise

    # Dynamics
    zero_crossing_rate_mean: Mapped[float | None] = mapped_column(Float)
    rms_mean: Mapped[float | None] = mapped_column(Float)

    # Rhythm. tempo_bpm is Fourier's own tempo estimate (librosa beat tracking, folded into
    # [60, 200]); Sononym's is in sononym_meta.bpm.
    tempo_bpm: Mapped[float | None] = mapped_column(Float)
    onset_rate_hz: Mapped[float | None] = mapped_column(Float)  # onsets per second

    # MFCCs (13 coefficients, stored as JSON array) — texture fingerprint for diversity
    mfcc_mean: Mapped[list | None] = mapped_column(JSONText)

    # ── Tier 2: later additions ────────────────────────────────────────────────────────────

    # Time from onset to peak RMS, in ms (10ms frame resolution).
    # snap/click ≈ 0–5ms; drums ≈ 5–25ms; swells ≈ 50–300ms.
    attack_time_ms: Mapped[float | None] = mapped_column(Float)

    # Time from peak RMS to −40dB below peak, in ms.
    # Tight electronic ≈ 100–400ms; acoustic ≈ 300–1000ms; roomy ≈ 1000ms+.
    decay_time_ms: Mapped[float | None] = mapped_column(Float)

    # Harmonic energy / total energy from HPSS decomposition. 1.0 = fully harmonic.
    # sub kick ≈ 0.60–0.80; snare ≈ 0.20–0.40; hat ≈ 0.05–0.25; pad ≈ 0.75–0.95.
    harmonic_percussive_ratio: Mapped[float | None] = mapped_column(Float)

    # max(chroma_mean) / mean(chroma_mean). High = focused on one pitch class.
    # tonal note ≈ 2.5–5.0; chord ≈ 1.5–2.5; noise/unpitched ≈ 1.0–1.3.
    chroma_concentration: Mapped[float | None] = mapped_column(Float)

    # ── Quality gate ─────────────────────────────────────────────────────────────────────

    # 1 if >0.1% of samples are at ±0.999 — audible digital clipping.
    is_clipped: Mapped[int | None] = mapped_column(Integer)

    # Absolute mean of the waveform (ideal = 0.0). >0.02 is an audible DC thump.
    dc_offset_ratio: Mapped[float | None] = mapped_column(Float)

    # ── Tier 3: CLAP semantic embeddings (deferred) ────────────────────────────────────────

    # 512-dim float32 packed as raw bytes (2048 bytes per sample). Use
    # analysis.clap_features.bytes_to_embedding() / embedding_to_bytes() to convert.
    clap_embedding: Mapped[bytes | None] = mapped_column(LargeBinary)
    clap_model: Mapped[str | None] = mapped_column(String)  # model id used

    # ── Musical key detection (Krumhansl-Schmuckler profile matching on chroma_cqt) ────────

    # Best-matching key from 24-key profile search.  Format: "<note> <mode>",
    # e.g. "C# minor", "F major".  NULL = not yet computed or unpitched sample.
    detected_key: Mapped[str | None] = mapped_column(String, nullable=True)

    # Pearson r of the best key match (0.0–1.0).  Values < 0.5 are suppressed
    # (detected_key set to NULL).  Stored to allow threshold tuning later.
    key_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)

    # ── Loop trim (non-destructive; applied at export time + UI preview) ───────────────────

    # Seconds from file start to the detected trim point.
    # NULL = not computed yet.  Set but zero-length silence = no trim warranted.
    # Applied in _convert_and_copy() during pack export when non-NULL, and used
    # to cap a waveform preview.
    trim_end_s: Mapped[float | None] = mapped_column(Float, nullable=True)

    # UTC timestamp of when loop-trim analysis was last run for this sample.
    # NULL trim_computed_at  → not yet processed.
    # trim_computed_at set, trim_end_s NULL → processed, < 50 ms trailing silence found.
    trim_computed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    # ── Sample-chain detection (analysis/events.py; `fourier tools analyze --only events`) ──────────────

    # Loud events separated by silence. A one-shot is 1; a velocity or note ladder
    # laid end to end in one file is many.
    n_events: Mapped[int | None] = mapped_column(Integer, nullable=True)
    # Coefficient of variation of event start spacing (None under 3 events).
    # ~0 = laid out on a fixed grid (a sample chain); larger = a performed phrase.
    event_regularity: Mapped[float | None] = mapped_column(Float, nullable=True)
    # 1 if every event is quieter than the one before (a delay/echo tail, not a chain).
    event_echo: Mapped[int | None] = mapped_column(Integer, nullable=True)
    events_computed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    # ── Fourier's own root note (pYIN; `fourier tools analyze --only own`) ──────────────────

    # MIDI pitch of a tonal one-shot's steady part by pYIN (packs/curate._detect_pitch), NULL
    # when it isn't clearly one pitch. Sononym's reading is sononym_meta.base_note.
    own_root_midi: Mapped[float | None] = mapped_column(Float, nullable=True)
    # When pYIN ran on it (own_root_midi NULL then means: no clear pitch). NULL: not yet, or
    # not a candidate (metadata/resolve.py ROOT_CANDIDATE_SQL).
    own_root_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    sample: Mapped[Sample] = relationship(back_populates="features")

    def to_dict(self) -> dict[str, Any]:
        return {
            # Tier 1
            "sub_weight": self.sub_weight,
            "transient_score": self.transient_score,
            "loop_confidence": self.loop_confidence,
            "is_pitched": self.is_pitched,
            "bpm_reliable": self.bpm_reliable,
            "has_timbral_norm": self.timbral_norm is not None,
            "derived_computed_at": self.derived_computed_at.isoformat() if self.derived_computed_at else None,
            # Tier 1 — later additions
            "spectral_balance": self.spectral_balance,
            "pitch_stability": self.pitch_stability,
            "attack_class": self.attack_class,
            "drum_subtype": self.drum_subtype,
            # Tier 2
            "spectral_centroid_mean": self.spectral_centroid_mean,
            "spectral_bandwidth_mean": self.spectral_bandwidth_mean,
            "spectral_rolloff_mean": self.spectral_rolloff_mean,
            "spectral_flatness_mean": self.spectral_flatness_mean,
            "zero_crossing_rate_mean": self.zero_crossing_rate_mean,
            "rms_mean": self.rms_mean,
            "tempo_bpm": self.tempo_bpm,
            "onset_rate_hz": self.onset_rate_hz,
            "has_mfcc": self.mfcc_mean is not None,
            # Tier 2 — later additions
            "attack_time_ms": self.attack_time_ms,
            "decay_time_ms": self.decay_time_ms,
            "harmonic_percussive_ratio": self.harmonic_percussive_ratio,
            "chroma_concentration": self.chroma_concentration,
            "has_clap": self.clap_embedding is not None,
            # Loop trim
            "trim_end_s": self.trim_end_s,
            "trim_computed_at": self.trim_computed_at.isoformat() if self.trim_computed_at else None,
            # Key detection
            "detected_key": self.detected_key,
            "key_confidence": self.key_confidence,
            # Sononym's tempo folded; Fourier's own root
            "sononym_bpm_folded": self.sononym_bpm_folded,
            "own_root_midi": self.own_root_midi,
        }


class DeviceProfile(Base):
    """Registered device configuration (loaded from YAML)."""

    __tablename__ = "device_profiles"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    device_id: Mapped[str] = mapped_column(String, unique=True, nullable=False)  # yaml 'id' field
    name: Mapped[str] = mapped_column(String, nullable=False)
    yaml_path: Mapped[str] = mapped_column(String, nullable=False)
    config_json: Mapped[dict] = mapped_column(JSONText, nullable=False)  # full parsed YAML
    last_used: Mapped[datetime | None] = mapped_column(DateTime)

    packs: Mapped[list[Pack]] = relationship(back_populates="device")


class Pack(Base):
    """A named collection of samples prepared for a specific device."""

    __tablename__ = "packs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String, nullable=False)
    device_id: Mapped[int | None] = mapped_column(ForeignKey("device_profiles.id"))
    description: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    export_path: Mapped[str | None] = mapped_column(String)  # where it was last exported
    skill_params: Mapped[dict | None] = mapped_column(JSONText)   # build parameters

    device: Mapped[DeviceProfile | None] = relationship(back_populates="packs")
    items: Mapped[list[PackItem]] = relationship(
        back_populates="pack", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<Pack '{self.name}' ({len(self.items)} items)>"


class PackItem(Base):
    """A sample within a pack, with device-specific destination path."""

    __tablename__ = "pack_items"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    pack_id: Mapped[int] = mapped_column(ForeignKey("packs.id"), nullable=False)
    sample_id: Mapped[int] = mapped_column(ForeignKey("samples.id"), nullable=False)
    dest_folder: Mapped[str | None] = mapped_column(String)   # relative folder within export
    dest_filename: Mapped[str | None] = mapped_column(String)  # renamed filename
    slot_name: Mapped[str | None] = mapped_column(String)      # e.g. "kick", "snare"
    cv_role: Mapped[str | None] = mapped_column(String)        # reserved
    sort_order: Mapped[int] = mapped_column(Integer, default=0)

    pack: Mapped[Pack] = relationship(back_populates="items")
    sample: Mapped[Sample] = relationship(back_populates="pack_items")

    __table_args__ = (UniqueConstraint("pack_id", "sample_id", name="uq_pack_sample"),)


# ---------------------------------------------------------------------------
# Provider metadata (docs/curation.md, "Sources and resolution"): every metadata source writes the
# same two shapes, so curation doesn't depend on one tool's schema. Rows are derived and
# rebuilt per provider (fourier/metadata/store.py); no foreign key, so pruning samples
# never trips over them.
# ---------------------------------------------------------------------------

class Label(Base):
    """A label a provider gives a sample: a class ("OneShot"), a category ("Perc Kicks"),
    a tag ("Kick"). rank keeps the provider's order; confidence is None when unknown."""

    __tablename__ = "labels"

    sample_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=False)
    provider: Mapped[str] = mapped_column(String, primary_key=True)     # "sononym", "ableton", ...
    kind: Mapped[str] = mapped_column(String, primary_key=True)         # "class" | "category" | "tag"
    label: Mapped[str] = mapped_column(String, primary_key=True)
    rank: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    confidence: Mapped[float | None] = mapped_column(Float)

    __table_args__ = (Index("ix_labels_provider_kind_label", "provider", "kind", "label"),)


class Descriptor(Base):
    """A measured value a provider gives a sample, in its own units (dB, BPM, MIDI note,
    0-1 timbre)."""

    __tablename__ = "descriptors"

    sample_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=False)
    provider: Mapped[str] = mapped_column(String, primary_key=True)
    name: Mapped[str] = mapped_column(String, primary_key=True)
    value: Mapped[float | None] = mapped_column(Float)

    __table_args__ = (Index("ix_descriptors_provider_name", "provider", "name"),)


# ---------------------------------------------------------------------------
# The library walk (fourier/ingest/walk.py): which samples a walk didn't find, and what each
# folder held when it was last listed. No foreign key, as above.
# ---------------------------------------------------------------------------

class MissingFile(Base):
    """A sample the last walk of its library folder didn't find there (moved, renamed,
    deleted or unreadable). Curation leaves it out (metadata/rows.py); a walk that finds the
    file again removes the row. Keyed by id and path, so a reused id never matches.
    `moved_to`: the path a walk found the same file at (importer._match_moves): an old row of
    a moved file, not a missing one (its analysis and ratings went with it; kept so a folder
    renamed back finds them)."""

    __tablename__ = "missing_files"

    sample_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=False)
    path: Mapped[str] = mapped_column(String, nullable=False)
    since: Mapped[int | None] = mapped_column(Integer)          # unix time it was first missed
    moved_to: Mapped[str | None] = mapped_column(String)        # where a walk found it moved


class WalkedDir(Base):
    """A folder as the walk last listed it: its modification time and what it held (sub
    folders, audio files, symlinks and the other files' extensions, as JSON). A later walk
    reuses the listing while the folder's time is unchanged, on file systems that update a
    folder's time whenever an entry is added, removed or renamed (platforms.dir_mtimes_reliable)."""

    __tablename__ = "walked_dirs"

    path: Mapped[str] = mapped_column(String, primary_key=True)
    mtime_ns: Mapped[int] = mapped_column(BigInteger, nullable=False)
    listing: Mapped[str] = mapped_column(Text, nullable=False)
