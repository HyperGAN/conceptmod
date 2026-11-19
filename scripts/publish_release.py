"""Publish the tested conceptmod wheel and sdist from a reviewed version."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tomllib


REPOSITORY = "HyperGAN/conceptmod"
CHECKSUMS = "SHA256SUMS.txt"


def version():
    value = tomllib.loads(Path("pyproject.toml").read_text())["project"]["version"]
    if not re.fullmatch(r"(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)", value):
        raise ValueError(f"Expected a stable major.minor.patch version, got {value!r}")
    return value


def gh(*args):
    return subprocess.check_output(["gh", *args], text=True)


def release_record(tag):
    filtered = gh("api", "--paginate", f"repos/{REPOSITORY}/releases?per_page=100",
                  "--jq", f"[.[] | select(.tag_name == {json.dumps(tag)})]")
    matches = [record for page in filtered.splitlines() for record in json.loads(page)]
    if len(matches) > 1:
        raise RuntimeError(f"Multiple GitHub release records for {tag}")
    return next(iter(matches), None)


def release_inputs(value, directory):
    distributions = [directory / f"conceptmod-{value}-py3-none-any.whl", directory / f"conceptmod-{value}.tar.gz"]
    hashes = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in distributions}
    checksum_file = directory / CHECKSUMS
    text = "".join(f"{digest}  {name}\n" for name, digest in hashes.items())
    return hashes, checksum_file, text


def verify_published(record, expected_names):
    public_assets = {a["name"] for a in record["assets"] if a["state"] == "uploaded"}
    if record["prerelease"] or not expected_names <= public_assets:
        raise RuntimeError(f"The existing release {record['tag_name']} requires review")


def plan(value, directory):
    tag = f"v{value}"
    hashes, checksums, text = release_inputs(value, directory)
    record = release_record(tag)
    if record and not record["draft"]:
        verify_published(record, {*hashes, CHECKSUMS})
        return "unchanged"
    if not (Path("docs/releases") / f"{value}.md").is_file():
        raise ValueError(f"Add docs/releases/{value}.md before publishing {tag}")
    checksums.write_text(text)
    return "candidate"


def current_main(sha):
    ref = gh("api", f"repos/{REPOSITORY}/git/ref/heads/main", "--jq", ".object.sha").strip()
    return ref == sha


def tag_commit(tag):
    ref = f"refs/tags/{tag}"
    refs = subprocess.check_output(
        ["git", "ls-remote", f"https://github.com/{REPOSITORY}.git", ref, f"{ref}^{{}}"], text=True
    )
    commits = {name: sha for sha, name in (row.split("\t") for row in refs.splitlines())}
    return commits.get(f"{ref}^{{}}", commits.get(ref))


def uploaded_digests(record, expected):
    existing = {}
    for asset in record["assets"]:
        name = asset["name"]
        if name not in expected or asset["state"] != "uploaded" or asset["digest"] != f"sha256:{expected[name]}":
            raise RuntimeError(f"Unexpected asset in draft {record['tag_name']}: {name}")
        existing[name] = expected[name]
    return existing


def publish(value, directory, sha):
    tag = f"v{value}"
    hashes, checksums, text = release_inputs(value, directory)
    actual = checksums.read_bytes()
    if actual != text.encode():
        raise RuntimeError("Distribution hashes differ from the tested checksum manifest")
    hashes[CHECKSUMS] = hashlib.sha256(actual).hexdigest()

    record = release_record(tag)
    if record and not record["draft"]:
        verify_published(record, hashes.keys())
        return "unchanged"
    if not current_main(sha):
        print(f"Superseded main revision {sha}")
        return "superseded"
    if tag_commit(tag) not in (None, sha):
        raise RuntimeError(f"Existing tag {tag} points elsewhere")

    notes = Path("docs/releases") / f"{value}.md"
    marker = f"<!-- conceptmod-release-source: {sha} -->"
    if record is None:
        title = f"Conceptmod {value}"
        gh("release", "create", tag, "--repo", REPOSITORY, "--draft",
           "--target", sha, "--title", title, "--notes", f"{notes.read_text()}\n\n{marker}")
        record = release_record(tag)
    if (not record or not record["draft"] or record["prerelease"]
            or record["target_commitish"] != sha or marker not in (record["body"] or "")):
        raise RuntimeError(f"Draft {tag} is not from the tested main revision {sha}")

    existing = uploaded_digests(record, hashes)
    missing = [str(directory / name) for name in hashes if name not in existing]
    if missing:
        gh("release", "upload", tag, "--repo", REPOSITORY, *missing)
    staged = release_record(tag)
    if not staged or staged["id"] != record["id"] or not staged["draft"] or staged["prerelease"]:
        raise RuntimeError(f"Draft {tag} changed while staging assets")
    if uploaded_digests(staged, hashes) != hashes:
        raise RuntimeError(f"GitHub did not receive all tested release assets for {tag}")
    if not current_main(sha):
        raise RuntimeError(f"Main advanced while staging {tag}. Review the draft before retrying.")
    current_tag = tag_commit(tag)
    if current_tag not in (None, sha):
        raise RuntimeError(f"Existing tag {tag} points elsewhere")
    if current_tag is None:
        gh("api", "--method", "POST", f"repos/{REPOSITORY}/git/refs",
           "-f", f"ref=refs/tags/{tag}", "-f", f"sha={sha}", "--silent")
        if tag_commit(tag) != sha:
            raise RuntimeError(f"Could not verify {tag} at the tested commit {sha}")

    gh("api", "--method", "PATCH", f"repos/{REPOSITORY}/releases/{staged['id']}",
       "-F", "draft=false", "--silent")
    published = release_record(tag)
    if (not published or published["id"] != staged["id"] or published["draft"]
            or published["prerelease"] or uploaded_digests(published, hashes) != hashes):
        raise RuntimeError(f"Published release {tag} did not match the tested artifacts")
    if tag_commit(tag) != sha:
        raise RuntimeError(f"Published tag {tag} does not point to {sha}")
    print(published["html_url"])
    return "published"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["version", "plan", "publish"])
    parser.add_argument("--dist", type=Path, default=Path("dist"))
    args = parser.parse_args()
    value = version()
    if args.action == "version":
        print(value)
    elif args.action == "plan":
        print(plan(value, args.dist))
    else:
        if (os.environ.get("GITHUB_REPOSITORY") != REPOSITORY or os.environ.get("GITHUB_REF") != "refs/heads/main"
                or os.environ.get("GITHUB_EVENT_NAME") != "push"):
            parser.error("Publishing requires a push to HyperGAN/conceptmod main")
        sha = os.environ.get("GITHUB_SHA", "")
        if not re.fullmatch(r"[0-9a-f]{40}", sha):
            parser.error("Set GITHUB_SHA to the tested commit")
        print(publish(value, args.dist, sha))


if __name__ == "__main__":
    main()
