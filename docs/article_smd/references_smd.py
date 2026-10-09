"""Bibliography in the SMD-UTM 2026 template format (MDPI-like):

journal:    AUTHOR, A.B.; AUTHOR, C.D. Title. *Journal* **Year**, *Volume*, pp. x–y, doi: ...
book:       AUTHOR, A.; AUTHOR, B. *Book title*; Publisher: City, Country, Year.
conference: AUTHOR, A.B. Title. In *Proceedings title*, Location, Country, Date; Publisher: City, Country, Year; pp. x–y.
website:    Site title. Disponibil online: URL (accesat la data de Day Month Year).

Markdown: *italic*, **bold**. The numbering follows the order of first citation in the text.
"""

REFS = {
    "gama2014": "GAMA, J.; ŽLIOBAITĖ, I.; BIFET, A.; PECHENIZKIY, M.; BOUCHACHIA, A. A survey on concept drift "
                "adaptation. *ACM Computing Surveys* **2014**, *46*(4), Article 44, pp. 1–37, doi: 10.1145/2523813.",
    "lu2019": "LU, J.; LIU, A.; DONG, F.; GU, F.; GAMA, J.; ZHANG, G. Learning under concept drift: A review. "
              "*IEEE Transactions on Knowledge and Data Engineering* **2019**, *31*(12), pp. 2346–2363, "
              "doi: 10.1109/TKDE.2018.2876857.",
    "gama2004": "GAMA, J.; MEDAS, P.; CASTILLO, G.; RODRIGUES, P. Learning with drift detection. In *Advances in "
                "Artificial Intelligence – SBIA 2004*, Proceedings of the 17th Brazilian Symposium on Artificial "
                "Intelligence, São Luis, Brazil, 29 September – 1 October 2004; Lecture Notes in Computer Science, "
                "vol. 3171; Springer: Berlin, Germany, 2004; pp. 286–295, doi: 10.1007/978-3-540-28645-5_29.",
    "bifet2007": "BIFET, A.; GAVALDÀ, R. Learning from time-changing data with adaptive windowing. In *Proceedings "
                 "of the 2007 SIAM International Conference on Data Mining*, Minneapolis, USA, 26–28 April 2007; "
                 "SIAM: Philadelphia, USA, 2007; pp. 443–448, doi: 10.1137/1.9781611972771.42.",
    "page1954": "PAGE, E.S. Continuous inspection schemes. *Biometrika* **1954**, *41*(1–2), pp. 100–115, "
                "doi: 10.1093/biomet/41.1-2.100.",
    "baier2023": "BAIER, L.; SCHLÖR, T.; SCHÖFFER, J.; KÜHL, N. Detecting concept drift with neural network model "
                 "uncertainty. In *Proceedings of the 56th Hawaii International Conference on System Sciences "
                 "(HICSS)*, Maui, USA, 3–6 January 2023. Disponibil online: "
                 "https://aisel.aisnet.org/hicss-56/da/big_data_and_analytics/6 (accesat la data de 9 octombrie 2026).",
    "kendall2017": "KENDALL, A.; GAL, Y. What uncertainties do we need in Bayesian deep learning for computer vision? "
                   "In *Advances in Neural Information Processing Systems 30*, Proceedings of NeurIPS 2017, Long Beach, "
                   "USA, 4–9 December 2017; Curran Associates: Red Hook, USA, 2017; pp. 5574–5584.",
    "depeweg2018": "DEPEWEG, S.; HERNÁNDEZ-LOBATO, J.M.; DOSHI-VELEZ, F.; UDLUFT, S. Decomposition of uncertainty "
                   "in Bayesian deep learning for efficient and risk-sensitive learning. In *Proceedings of the 35th "
                   "International Conference on Machine Learning*, Stockholm, Sweden, 10–15 July 2018; Proceedings of "
                   "Machine Learning Research, vol. 80; PMLR, 2018; pp. 1184–1193.",
    "hullermeier2021": "HÜLLERMEIER, E.; WAEGEMAN, W. Aleatoric and epistemic uncertainty in machine learning: an "
                       "introduction to concepts and methods. *Machine Learning* **2021**, *110*, pp. 457–506, "
                       "doi: 10.1007/s10994-021-05946-3.",
    "gal2016": "GAL, Y.; GHAHRAMANI, Z. Dropout as a Bayesian approximation: Representing model uncertainty in deep "
               "learning. In *Proceedings of the 33rd International Conference on Machine Learning*, New York, USA, "
               "19–24 June 2016; Proceedings of Machine Learning Research, vol. 48; PMLR, 2016; pp. 1050–1059.",
    "lakshminarayanan2017": "LAKSHMINARAYANAN, B.; PRITZEL, A.; BLUNDELL, C. Simple and scalable predictive "
                            "uncertainty estimation using deep ensembles. In *Advances in Neural Information "
                            "Processing Systems 30*, Proceedings of NeurIPS 2017, Long Beach, USA, 4–9 December 2017; "
                            "Curran Associates: Red Hook, USA, 2017; pp. 6402–6413.",
    "breiman2001": "BREIMAN, L. Random forests. *Machine Learning* **2001**, *45*(1), pp. 5–32, "
                   "doi: 10.1023/A:1010933404324.",
    "shaker2020": "SHAKER, M.H.; HÜLLERMEIER, E. Aleatoric and epistemic uncertainty with random forests. In "
                  "*Advances in Intelligent Data Analysis XVIII*, Proceedings of IDA 2020, Konstanz, Germany, "
                  "27–29 April 2020; Lecture Notes in Computer Science, vol. 12080; Springer: Cham, Switzerland, "
                  "2020; pp. 444–456, doi: 10.1007/978-3-030-44584-3_35.",
    "adams2007": "ADAMS, R.P.; MACKAY, D.J.C. Bayesian online changepoint detection. *arXiv preprint* **2007**, "
                 "arXiv:0710.3742.",
    "dempster1977": "DEMPSTER, A.P.; LAIRD, N.M.; RUBIN, D.B. Maximum likelihood from incomplete data via the EM "
                    "algorithm. *Journal of the Royal Statistical Society: Series B (Methodological)* **1977**, "
                    "*39*(1), pp. 1–38, doi: 10.1111/j.2517-6161.1977.tb01600.x.",
    "rabanser2019": "RABANSER, S.; GÜNNEMANN, S.; LIPTON, Z.C. Failing loudly: An empirical study of methods for "
                    "detecting dataset shift. In *Advances in Neural Information Processing Systems 32*, Proceedings "
                    "of NeurIPS 2019, Vancouver, Canada, 8–14 December 2019; Curran Associates: Red Hook, USA, 2019.",
    "lundberg2017": "LUNDBERG, S.M.; LEE, S.-I. A unified approach to interpreting model predictions. In *Advances "
                    "in Neural Information Processing Systems 30*, Proceedings of NeurIPS 2017, Long Beach, USA, "
                    "4–9 December 2017; Curran Associates: Red Hook, USA, 2017; pp. 4765–4774.",
    "ribeiro2016": "RIBEIRO, M.T.; SINGH, S.; GUESTRIN, C. „Why should I trust you?”: Explaining the predictions of "
                   "any classifier. In *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge "
                   "Discovery and Data Mining*, San Francisco, USA, 13–17 August 2016; ACM: New York, USA, 2016; "
                   "pp. 1135–1144, doi: 10.1145/2939672.2939778.",
    "yeh2009": "YEH, I.-C.; LIEN, C.-H. The comparisons of data mining techniques for the predictive accuracy of "
               "probability of default of credit card clients. *Expert Systems with Applications* **2009**, *36*(2), "
               "pp. 2473–2480, doi: 10.1016/j.eswa.2007.12.020.",
    "ibm2019": "IBM. Telco Customer Churn (set de date). Disponibil online: "
               "https://github.com/IBM/telco-customer-churn-on-icp4d (accesat la data de 9 octombrie 2026).",
    "harries1999": "HARRIES, M. *SPLICE-2 Comparative Evaluation: Electricity Pricing*; Technical Report; "
                   "University of New South Wales: Sydney, Australia, 1999.",
    "street2001": "STREET, W.N.; KIM, Y. A streaming ensemble algorithm (SEA) for large-scale classification. In "
                  "*Proceedings of the 7th ACM SIGKDD International Conference on Knowledge Discovery and Data "
                  "Mining*, San Francisco, USA, 26–29 August 2001; ACM: New York, USA, 2001; pp. 377–382, "
                  "doi: 10.1145/502512.502568.",
    "montiel2021": "MONTIEL, J.; HALFORD, M.; MASTELINI, S.M.; BOLMIER, G.; SOURTY, R.; VAYSSE, R.; et al. River: "
                   "machine learning for streaming data in Python. *Journal of Machine Learning Research* **2021**, "
                   "*22*(110), pp. 1–8.",
    "hein2019": "HEIN, M.; ANDRIUSHCHENKO, M.; BITTERWOLF, J. Why ReLU networks yield high-confidence predictions far "
                "away from the training data and how to mitigate the problem. In *Proceedings of the IEEE/CVF "
                "Conference on Computer Vision and Pattern Recognition (CVPR)*, Long Beach, USA, 16–20 June 2019; "
                "IEEE: Piscataway, USA, 2019; pp. 41–50, doi: 10.1109/CVPR.2019.00013.",
    "holm1979": "HOLM, S. A simple sequentially rejective multiple test procedure. *Scandinavian Journal of "
                "Statistics* **1979**, *6*(2), pp. 65–70.",
    "schuirmann1987": "SCHUIRMANN, D.J. A comparison of the two one-sided tests procedure and the power approach for "
                      "assessing the equivalence of average bioavailability. *Journal of Pharmacokinetics and "
                      "Biopharmaceutics* **1987**, *15*(6), pp. 657–680, doi: 10.1007/BF01068419.",
    "kunsch1989": "KÜNSCH, H.R. The jackknife and the bootstrap for general stationary observations. *The Annals of "
                  "Statistics* **1989**, *17*(3), pp. 1217–1241, doi: 10.1214/aos/1176347265.",
    "dersimonian1986": "DERSIMONIAN, R.; LAIRD, N. Meta-analysis in clinical trials. *Controlled Clinical Trials* "
                       "**1986**, *7*(3), pp. 177–188, doi: 10.1016/0197-2456(86)90046-2.",
    "wimmer2023": "WIMMER, L.; SALE, Y.; HOFMAN, P.; BISCHL, B.; HÜLLERMEIER, E. Quantifying aleatoric and epistemic "
                  "uncertainty in machine learning: Are conditional entropy and mutual information appropriate "
                  "measures? In *Proceedings of the 39th Conference on Uncertainty in Artificial Intelligence*, "
                  "Pittsburgh, USA, 31 July – 4 August 2023; Proceedings of Machine Learning Research, vol. 216; "
                  "PMLR, 2023; pp. 2282–2292.",
}
