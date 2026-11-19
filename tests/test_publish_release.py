import hashlib
import importlib.util
from pathlib import Path
from unittest.mock import patch

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "publish_release.py"
_SPEC = importlib.util.spec_from_file_location("conceptmod_publish_release", _SCRIPT)
release = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(release)


VERSION = "3.1.1"
TAG = f"v{VERSION}"
SHA = "a" * 40
RELEASE_ID = 1234


@pytest.fixture
def tested_dist(tmp_path):
    (tmp_path / f"conceptmod-{VERSION}-py3-none-any.whl").write_bytes(b"tested wheel")
    (tmp_path / f"conceptmod-{VERSION}.tar.gz").write_bytes(b"tested source")
    digests, path, manifest = release.release_inputs(VERSION, tmp_path)
    path.write_text(manifest)
    digests[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return tmp_path, digests


def record(digests, *, draft):
    return {
        "id": RELEASE_ID, "tag_name": TAG, "draft": draft, "prerelease": False,
        "target_commitish": SHA, "body": f"<!-- conceptmod-release-source: {SHA} -->",
        "html_url": f"https://github.com/HyperGAN/conceptmod/releases/tag/{TAG}",
        "assets": [
            {"name": name, "state": "uploaded", "digest": f"sha256:{digest}"}
            for name, digest in digests.items()
        ],
    }


def test_existing_public_release_skips_writes(tested_dist):
    directory, digests = tested_dist
    with patch.object(release, "release_record", return_value=record(digests, draft=False)):
        with patch.object(release, "gh") as gh:
            assert release.plan(VERSION, directory) == "unchanged"
            assert release.publish(VERSION, directory, SHA) == "unchanged"
    gh.assert_not_called()


def test_new_release_finishes_only_after_all_hashes_match(tested_dist):
    directory, digests = tested_dist
    records = [None, record({}, draft=True), record(digests, draft=True), record(digests, draft=False)]
    with patch.object(release, "release_record", side_effect=records):
        with patch.object(release, "current_main", return_value=True):
            with patch.object(release, "tag_commit", side_effect=[None, None, SHA, SHA]):
                with patch.object(release, "gh") as gh:
                    assert release.publish(VERSION, directory, SHA) == "published"
    create, upload, bind, publish = [call.args for call in gh.call_args_list]
    assert create[:3] == ("release", "create", TAG)
    assert "--draft" in create and create[create.index("--target") + 1] == SHA
    assert upload == ("release", "upload", TAG, "--repo", release.REPOSITORY,
                      *(str(directory / name) for name in digests))
    assert bind == ("api", "--method", "POST", f"repos/{release.REPOSITORY}/git/refs",
                    "-f", f"ref=refs/tags/{TAG}", "-f", f"sha={SHA}", "--silent")
    assert publish == ("api", "--method", "PATCH", f"repos/{release.REPOSITORY}/releases/{RELEASE_ID}",
                       "-F", "draft=false", "--silent")


def test_retry_only_uploads_missing_files_when_draft_matches(tested_dist):
    directory, digests = tested_dist
    wheel = f"conceptmod-{VERSION}-py3-none-any.whl"
    records = [record({wheel: digests[wheel]}, draft=True), record(digests, draft=True),
               record(digests, draft=False)]
    with patch.object(release, "release_record", side_effect=records):
        with patch.object(release, "current_main", return_value=True):
            with patch.object(release, "tag_commit", return_value=SHA):
                with patch.object(release, "gh") as gh:
                    assert release.publish(VERSION, directory, SHA) == "published"
    upload, publish = [call.args for call in gh.call_args_list]
    assert upload[5:] == tuple(str(directory / name) for name in digests if name != wheel)
    assert publish[:3] == ("api", "--method", "PATCH")


@pytest.mark.parametrize("late_tag", [None, "b" * 40])
def test_unverified_created_tag_leaves_the_release_draft(tested_dist, late_tag):
    directory, digests = tested_dist
    with patch.object(release, "release_record", return_value=record(digests, draft=True)):
        with patch.object(release, "current_main", return_value=True):
            with patch.object(release, "tag_commit", side_effect=[None, None, late_tag]):
                with patch.object(release, "gh") as gh:
                    with pytest.raises(RuntimeError, match="Could not verify"):
                        release.publish(VERSION, directory, SHA)
    assert gh.call_count == 1
    assert gh.call_args.args[:3] == ("api", "--method", "POST")


def test_replaced_draft_is_not_published(tested_dist):
    directory, digests = tested_dist
    draft = record(digests, draft=True)
    records = [draft, {**draft, "id": RELEASE_ID + 1}]
    with patch.object(release, "release_record", side_effect=records):
        with patch.object(release, "current_main", return_value=True):
            with patch.object(release, "tag_commit", return_value=SHA):
                with patch.object(release, "gh") as gh:
                    with pytest.raises(RuntimeError, match="changed while staging"):
                        release.publish(VERSION, directory, SHA)
    gh.assert_not_called()


def test_mismatched_draft_never_publishes(tested_dist):
    directory, digests = tested_dist
    name = next(iter(digests))
    with patch.object(release, "release_record", return_value=record({name: "0" * 64}, draft=True)):
        with patch.object(release, "current_main", return_value=True):
            with patch.object(release, "tag_commit", return_value=None):
                with patch.object(release, "gh") as gh:
                    with pytest.raises(RuntimeError, match="Unexpected asset"):
                        release.publish(VERSION, directory, SHA)
    gh.assert_not_called()


def test_stale_main_and_conflicting_tag_do_not_write(tested_dist):
    directory, _ = tested_dist
    with patch.object(release, "release_record", return_value=None):
        with patch.object(release, "gh") as gh:
            with patch.object(release, "current_main", return_value=False):
                assert release.publish(VERSION, directory, SHA) == "superseded"
            with patch.object(release, "current_main", return_value=True):
                with patch.object(release, "tag_commit", return_value="b" * 40):
                    with pytest.raises(RuntimeError, match="points elsewhere"):
                        release.publish(VERSION, directory, SHA)
    gh.assert_not_called()


def test_tampered_distribution_fails_before_provider_access(tested_dist):
    directory, _ = tested_dist
    (directory / f"conceptmod-{VERSION}.tar.gz").write_bytes(b"wrong source")
    with patch.object(release, "release_record") as read_release:
        with pytest.raises(RuntimeError, match="differ from the tested checksum manifest"):
            release.publish(VERSION, directory, SHA)
    read_release.assert_not_called()
