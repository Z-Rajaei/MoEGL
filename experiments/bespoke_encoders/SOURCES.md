# Bespoke model-to-graph encoders used for the LoC comparison

Only the files that implement the *model-to-graph encoding* were copied; training,
generation and plotting code of the original projects is deliberately excluded.

| Key      | Publication                                    | Origin                                                                                     | Files copied                                                                         |
|----------|------------------------------------------------|--------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------|
| lopez    | Lopez & Cuadrado, MODELS'21 / TSE'22 (TCRMG-GNN) | https://github.com/Antolin1/TCRMG-GNN (release 1.0.0)                                        | java/GraphGeneration/src/main/java/gg/core/{Parser,GraphModel,IMetaFilter,MetaFilterLiterals,MetaFilterNames}.java, gg/loaders/{RdsFull,YakinduFullLoader}.java, gg/main/GenerateRealGraphs{Ecore,RDS,Yakindu}.java, python/json2graph.py |
| rahimi   | Rahimi et al., MODELS-C'23 (NetGAN for models)  | authors' project (netgan/encoder.py)                                                        | netgan/encoder.py                                                                    |
| miranda  | Miranda et al., SAC'24 / ECMFA'24               | https://github.com/NaoMod/Support-ML-Relations-Model-Views (main, fetched 2026-09-28)       | Python/utils/to_graph.py, Python/utils/encoders.py, Python/modeling/metamodels.py    |

LoC are counted with `cloc` (code lines only, comments and blanks excluded); see `../scripts/count_loc.py`.
