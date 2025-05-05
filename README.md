# Arawan_Phylogeny
Repository accompanying the "Bayesian phylogenetic study of the Arawan language family" paper.  
The data is taken from Gerardi et al. (2022) and can be found [here](https://github.com/tupian-language-resources/kahd). This also has the expert cognacy judgements used to assess the quality of the automated cognate detection described in the paper.

The [Trees](/Trees/) folder contains the settings and the output of tree runs conducted. [data](/data/) contains statistics on both datasets used in the study (FULL and REDUCED). The underlying datasets can be found in [raw](/raw/). [etc](/etc/) contains the restrictions on doculects and concepts used in the build_nexus.py script.



## Generating Nexus files

Build the nexus file with the following command; this will use zero for ascertainment bias
when a concept is not observed, and rename/filter doculects and parameters. The experiment
will be called "base" and the output will be written to the "output" directory.

```bash
python .\build_nexus.py -z -l DOCULECT -d .\etc\arawan.txt -p .\etc\concepts.txt .\raw\arawan.20230630.tsv base
```

## Sources
Gerardi, F.F., Aragon, C.C. and Reichert, S. (2022) ‘KAHD: Katukinan-Arawan-Harakmbut Database (Pre-release)’, Journal of Open Humanities Data, 8(0), p. 18. Available at: https://doi.org/10.5334/johd.80.

