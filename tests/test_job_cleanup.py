"""Old uploads are deleted, including folders the server can't reload after a restart."""
import os
import time

import pytest

import api.main as server


@pytest.fixture
def jobs_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "JOBS_DIR", tmp_path)
    monkeypatch.setattr(server, "jobs", {})
    return tmp_path


def folder(root, name, hours_old):
    d = root / name
    (d / "input").mkdir(parents=True)
    t = time.time() - hours_old * 3600
    os.utime(d, (t, t))
    return d


def test_old_unknown_folders_are_removed_recent_ones_kept(jobs_dir):
    ttl = server.JOB_TTL_HOURS
    stale = folder(jobs_dir, "stale_orphan", ttl + 1)    # an interrupted upload from long ago
    fresh = folder(jobs_dir, "fresh_orphan", 0)          # just created: maybe an upload starting right now
    finished_old = folder(jobs_dir, "old_job", ttl + 1)  # a finished job past its time
    current = folder(jobs_dir, "current_job", ttl + 1)   # old folder, but its job is recent and known
    server.jobs["old_job"] = server.Job(id="old_job", models=["segresnet"], status="done",
                                        created=time.time() - (ttl + 1) * 3600)
    server.jobs["current_job"] = server.Job(id="current_job", models=["segresnet"], status="done")

    server.clean_old_jobs()

    assert not stale.exists() and not finished_old.exists()
    assert fresh.exists() and current.exists()
    assert set(server.jobs) == {"current_job"}
