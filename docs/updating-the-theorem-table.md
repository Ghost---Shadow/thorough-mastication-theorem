# Updating the theorem table when the library grows

[theorem-table.md](theorem-table.md) lists every theorem of the
[squiggly line library](https://github.com/Ghost---Shadow/squiggly-line-theorem) with a
verdict: does it help to say what happens when a batch is trained to convergence, and why.
This page says how to bring the table up to date after the library gains theorems.

## The files

| File | Written by | What it holds |
|---|---|---|
| `docs/theorem-table.md` | the script | the table; never edit it by hand |
| `docs/verdicts.json` | a person or an agent | the "helps" and "why" cells |
| `docs/seen-theorems.txt` | the script | the names of the theorems that were there at the last pass |
| `scripts/make_theorem_table.py` | | makes the table from the library and the two files above |
| `lean/lakefile.toml` | a person | the commit of the library this repository is pinned to |

The theorem and "what it says" columns are read from the Lean sources of the library by its
own search tool (`scripts/find_theorem.py` there). So the script needs a checkout of the
library. It looks in `--library PATH`, then in the variable `SQUIGGLY_LINE_REPO`, then next to
this repository (`../squiggly-line-theorem` or `../squiggly-line-conjecture`).

## The steps

1. **Bring the library checkout to the commit you want**, for example `git pull` in it.

2. **Regenerate the table.**

   ```bash
   python scripts/make_theorem_table.py --write
   ```

   Every theorem that is not in `docs/seen-theorems.txt` now shows as "not judged", even in a
   file that already has a verdict. If the script says that `verdicts.json` names a theorem
   the library does not have, that theorem was renamed or removed: fix the name in
   `verdicts.json` (the library's `lean/RENAMES-theorems.txt` has old and new names).

3. **List what is not judged.**

   ```bash
   python scripts/make_theorem_table.py --new
   ```

   Each line is a theorem, its file, and the first sentence of its doc comment.

4. **Judge them.** Read each new theorem. In the library:

   ```bash
   python scripts/find_theorem.py show NAME
   ```

   For a theorem that helps, add an entry to `"theorems"` in `docs/verdicts.json`:

   ```json
   "name_of_the_theorem": ["yes", "One or two sentences on why it helps."]
   ```

   The first value is `yes`, `partly` or `no`. A theorem with no entry of its own takes the
   verdict of its file, and failing that of its folder, from `"groups"`. For a new file, add
   it to a group or make a new group:

   ```json
   {"helps": "no", "why": "File: what the file is about, and why that does not bear on a mastication.",
    "files": ["Supporting/Training/NewFile"]}
   ```

   A file path is written without `lean/SquigglyLine/` and without `.lean`. An entry ending
   in `/*` covers a folder, and an exact file wins over a folder. Start the reason of a group
   with "File:". It tells the reader that the theorems of that file were judged by the file's
   subject and not one by one.

5. **Record them as seen, and regenerate.**

   ```bash
   python scripts/make_theorem_table.py --mark-seen
   python scripts/make_theorem_table.py --write
   ```

   Do this only after step 4. `--mark-seen` is what turns "not judged" into the verdict of
   the file, so running it early hides the new theorems again.

6. **Move the pin.** Put the full commit hash of the library checkout into `rev` in
   `lean/lakefile.toml`. The table records the commit it was made from, and the CI check
   compares against the pinned commit, so the two have to agree. Then check that the proofs
   still compile against the new commit (the README says how).

7. **Check and commit.**

   ```bash
   python scripts/make_theorem_table.py --check
   ```

   Commit `docs/theorem-table.md`, `docs/verdicts.json`, `docs/seen-theorems.txt` and
   `lean/lakefile.toml` together.

## What the verdicts mean

- **yes**: the theorem says something about a mastication directly, and the reason says what.
- **partly**: it supports a "yes" row or covers a special case.
- **no**: it is about something else.
- **not judged**: nobody has looked at it. This is either a theorem added since the last
  pass, or a file marked for a later reading.

A new theorem can also call for a new proof here. If a "yes" row makes one of the statements
in [proofs.md](proofs.md) stronger or shorter, say so in its reason.

## What checks this

`.github/workflows/ci.yml` has a job that runs `--check` against the library at the pinned
commit. It is switched off while the repository is private.
