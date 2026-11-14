# Releasing

A release is a git tag. Pushing `vX.Y.Z` makes GitHub build every distribution, test each one
as installed, and upload them to PyPI. Nothing is built or uploaded from a workstation. This page
covers the routine, the one-time setup it relies on, what to do when a run fails, and what to
keep current as Python and the tools move on.

## What a release publishes

| Distribution | Built by | Tested by |
|---|---|---|
| **40 binary wheels**: CPython 3.11–3.15 × Linux x86_64 and aarch64 (glibc and musl), macOS arm64 and x86_64, Windows AMD64 and ARM64 | `cibuildwheel` with `LAGO_REQUIRE_COMPILED=1`: a failed compile fails the build | the test suite with every compiled module asserted present (`tests/core/test_build.py`), then `benchmarks/compare_backends.py --require-compiled` (compiled and pure paths bit-identical) |
| **Pure-Python wheel** `py3-none-any`: what pip installs where no binary wheel matches (PyPy, Pyodide, other architectures) | `LAGO_PURE=1 python -m build --wheel` | the test suite on CPython 3.11, the latest CPython and PyPy 3.11 |
| **sdist**, with the C that Cython generated | `python -m build --sdist` | installed once without a C compiler (must give pure Python) and once with one (must give compiled) |

Where each part is configured:
- `.github/workflows/wheels.yml`: the jobs and their order. The publish job runs only after every build and test job has passed.
- `pyproject.toml`, `[tool.cibuildwheel]`: which binary wheels are built and how they are tested.
- `setup.py`: the `LAGO_PURE` and `LAGO_REQUIRE_COMPILED` switches.

The design behind this is in [ARCHITECTURE.md](ARCHITECTURE.md) §8.

## Why not `python -m build` and `twine upload`

That was the process up to 1.2.0, and it is why 1.2.0 shipped compiled wheels for one platform
only:
- A local build produces one wheel, for the machine and Python version it runs on.
- Windows wheels cannot be built on a Mac.
- On Apple Silicon, Linux x86_64 wheels need emulated Docker and are very slow.
- `build/` directories left over from earlier builds are packaged as they are, so a local wheel can carry binaries older than its sources.
- A token upload carries no provenance.

**Never upload a file built on a workstation.** A local wheel is for local testing only (see the end of this page).

## One-time setup

1. **PyPI trusted publisher.** GitHub, not a person, uploads the files, and PyPI must be told to
   accept uploads from this workflow. A "publisher" is not a co-owner: it is a rule in the
   project settings that only an owner can add.
   - Where: https://pypi.org/manage/project/dcd-lago/settings/publishing/ → add a GitHub publisher.
   - Owner: `fondationsahar`
   - Repository: `dynamic_community_detection`
   - Workflow: `wheels.yml`
   - Environment: `pypi`

   On each release GitHub then gets a single-use token, valid for about 15 minutes; no password is
   stored anywhere. The same rule lets PyPI show, on each file's page, the commit and workflow
   that built it (its provenance). PyPI has no API for this setting, so this step needs the browser.
2. **The `pypi` GitHub environment** is created automatically the first time the publish job
   runs, with no protection, so pushing a tag publishes as soon as everything is green.
   Optionally, in the repository's Settings → Environments → `pypi`:
   - add yourself as a required reviewer, so that each publish waits for an approval;
   - under "Deployment branches and tags", restrict it to the tag pattern `v*`.
3. **TestPyPI (optional),** for rehearsals. It needs its own account on test.pypi.org. Add a
   pending publisher at https://test.pypi.org/manage/account/publishing/ with project name
   `dcd-lago`, the same owner, repository and workflow as step 1, and environment `testpypi`.
4. **After the first release published this way,** revoke the old PyPI API token (pypi.org →
   Account settings → API tokens) and remove it from `~/.pypirc`. Nothing uses it any more.

## Releasing a version

During development, the next version is set in `pyproject.toml` (`version = "X.Y.Z"`) and
described in `CHANGELOG.md` under `## X.Y.Z — unreleased`.

1. Run the checks locally (a compiled build needs a C compiler; `pip install -e ".[dev]"` installs the rest):
   ```bash
   python setup.py build_ext --inplace
   python -m pytest tests -q
   python benchmarks/compare_backends.py -n 1000 --require-compiled
   ```
2. Date the changelog (macOS `sed`; on Linux, `sed -i` without the `''`):
   ```bash
   sed -i '' "s/^## X.Y.Z — unreleased$/## X.Y.Z — $(date +%F)/" CHANGELOG.md
   ```
3. Commit it:
   ```bash
   git commit -am "release: X.Y.Z"
   ```
4. Push `main`:
   ```bash
   git push origin main
   ```
5. Tag the release. The tag must be `v` followed by exactly the version in `pyproject.toml`; the
   publish job refuses to run otherwise.
   ```bash
   git tag -a vX.Y.Z -m "X.Y.Z"
   ```
6. Push the tag, which starts the release run. It takes about 30–60 minutes.
   ```bash
   git push origin vX.Y.Z
   ```
7. Check what PyPI received. Expect 42 files (40 binary wheels, the pure wheel and the sdist) as
   long as the build list is unchanged.
   ```bash
   curl -s https://pypi.org/simple/dcd-lago/ | grep -o 'dcd_lago-X\.Y\.Z[^"#<]*' | sort -u | wc -l
   ```
8. Check that a machine with no compiler gets the compiled build. It should print `compiled cython`:
   ```bash
   docker run --rm python:3.12-slim sh -c "pip install -q --only-binary :all: dcd-lago==X.Y.Z && python -c 'import lago.accel as a; print(a.core_name(), a.backend_name)'"
   ```

## When a release run fails

- **A build or test job failed.** Nothing was published: the publish job only runs once all of
  them have passed. Read the failed job's log in the repository's Actions tab (or run
  `gh run view --log-failed` if the GitHub CLI is installed). Fix the problem on `main`, then
  move the tag, which is only safe because nothing was published:
  ```bash
  git tag -d vX.Y.Z && git push origin :refs/tags/vX.Y.Z
  ```
  Then repeat steps 5 and 6.
- **The publish job failed after uploading some files.** PyPI never accepts the same file name
  twice, not even after the file is deleted. Release the next patch version instead.

## Testing before tagging (optional)

- A pull request that touches `setup.py`, `pyproject.toml`, `MANIFEST.in`, a `.pyx`/`.pxd` file
  or the workflow runs every build and test job, but never publishes.
- Changes to the `.py` sources alone do not trigger the workflow. To build and test `main` on
  every platform before tagging: Actions → Wheels → **Run workflow** on `main`, with "Publish to
  TestPyPI" left unticked.
- To rehearse the upload as well, tick "Publish to TestPyPI" (this needs setup step 3). TestPyPI
  also refuses a version twice, so a rehearsal can be done only once per version.

## Keeping it current

- **A new CPython every October.** Add `"cp3NN-*"` to `build` in `[tool.cibuildwheel]`, add the
  classifier in `[project]`, and update cibuildwheel (and Cython, if needed) to versions that
  support it. The free-threaded skip pattern `cp31?t-*` covers versions 3.10–3.19.
- **The pinned actions.** Each `uses:` line in the workflow pins a commit, with its version in a
  comment. To update one, take the commit of the new tag. For an annotated tag, that is the
  `^{}` line:
  ```bash
  git ls-remote https://github.com/OWNER/REPO 'refs/tags/vX.Y.Z' 'refs/tags/vX.Y.Z^{}'
  ```
  Prefer versions released at least two weeks earlier. The same applies to the
  `pypa/cibuildwheel` version.
- **Retired runner images.** `macos-15-intel` is GitHub's last Intel macOS image. When it
  goes, either drop Intel Mac wheels (the pure wheel covers those Macs) or cross-build them on
  the arm64 runner with `archs = ["x86_64"]` under `[tool.cibuildwheel.macos]`.
- **The list of compiled modules.** `setup.py` (`COMPILED_MODULES`) is the source.
  `tests/core/test_build.py` mirrors it (its `test_matches_setup` fails if they diverge), and the
  workflow's "Check what each distribution carries" step expects **12** generated C files in the
  sdist. Update that number whenever a module is added to or removed from the compiled build.
- **Cython.** The build requirement is `Cython>=3.1,<4`; revisit the upper bound when Cython 4
  is released.
- **No 32-bit x86, no free-threaded builds.** These are deliberate (see the comment in
  `[tool.cibuildwheel]`). Lifting either needs the bit-identity check to pass on that platform first.

## Local builds, for development only

- `python setup.py build_ext --inplace` puts a `.so` beside each compiled `.py`. Python prefers
  the `.so`, so **rebuild after editing a compiled module**, or the old binary keeps running.
  `LAGO_CORE=python` and `LAGO_ACCEL=python` force the `.py` sources;
  `lago.accel.core_name()` and `lago.accel.backend_name` say what is running.
- `LAGO_PURE=1 python -m build --wheel` builds the pure wheel, and `LAGO_REQUIRE_COMPILED=1`
  turns a failed compile into an error. Delete `build/` before building a wheel locally:
  leftover binaries in it get packaged.
