# Third party notices and implementation ancestry

Project-authored contributions are released under the root [MIT license](LICENSE). This does not replace third-party copyright notices. Preserve the notices below when redistributing the corresponding code or substantial portions of it.

## TGL derived components

The inherited runtimes under [the clock-weight comparison](experiments/clock_weights/inherited) and [the weight-reassignment comparison](experiments/weight_reassignment/inherited) contain modified components derived from the [TGL companion repository](https://github.com/dess-mannheim/tgl) for *What Do Temporal Graph Learning Models Learn?* by Abigail J. Hayes, Tobias Schumacher and Markus Strohmaier.

- Retained TGL ancestor: commit `c7b130f93078aa94155856c3d8990a6ac70bc975`.
- License: MIT, copyright (c) 2025 Data Science in the Economic and Social Sciences.
- Full retained notice: [licenses/inherited-MIT.txt](licenses/inherited-MIT.txt).
- Retained notice SHA-256: `3f33c46cd370d850780e1a5c19fcdad60e9eb58b6245bb4993539b18b924a4c8`.
- Official license reference: [TGL LICENSE](https://github.com/dess-mannheim/tgl/blob/main/LICENSE), inspected on 2026-11-14 and consistent with the retained notice.

The retained ancestor identifies the historical TGL source, not an assertion that every inherited file is unchanged upstream code. The project subsequently added query-feature interventions, evaluation controls and functional state handling. In particular, `inherited/tgn_host.py` implements the project's functional atomic-snapshot adapter; it is not an unmodified released TGN implementation. Files under `inherited/runtime/` include both adapted upstream components and project additions. Source headers and provenance distinguish their roles.

## DyGLib ancestry

TGL attributes parts of its model and training implementation to [DyGLib](https://github.com/yule-BUAA/DyGLib). The inherited `runtime/train/models/MemoryModel.py` and `runtime/train/models/modules.py` retain that attribution.

- License: MIT, copyright (c) 2023 Yu Le.
- Full notice: [licenses/DyGLib-MIT.txt](licenses/DyGLib-MIT.txt).
- Official license reference: [DyGLib LICENSE](https://github.com/yule-BUAA/DyGLib/blob/master/LICENSE), inspected on 2026-11-14.

An exact separate DyGLib ancestor commit is not established by the retained TGL provenance. The current official license inspection confirms the MIT notice; it does not supply that missing historical version pin. Accordingly, this repository describes its host as modified TGL/DyGLib-derived components, not an exact reproduction of a separately pinned DyGLib release.

## Method attribution and external dependencies

The recurrence-conditioned query adapters and functional training workflow are project modifications. Methodological relationships to TGN, KEAT and other research do not mean that this repository is the released KEAT-TGN or KEAT-DyGFormer code. Cite the respective research when discussing those relationships; do not replace implementation provenance with a paper citation.

Python packages installed through the environment requirements retain their respective licenses. Their distributions are not vendored by this repository. The synthetic rerun workflow does not redistribute CollegeMsg, flight records or other external real-world datasets.

## Redistribution scope

The TGL and DyGLib licenses inspected here are both MIT. Their required copyright and permission notices are included above and in the accompanying license files; no incompatible license was identified in this inherited source chain. This statement is a source-and-notice inventory, not a guarantee about code or data added later. Preserve existing attributions and review any additional vendored dependency before publishing it.
