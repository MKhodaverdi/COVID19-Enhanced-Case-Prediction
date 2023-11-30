# COVID19-Enhanced-Case-Prediction

# Enhanced SARS-CoV-2 Case Prediction Using Public Health Data and Machine Learning Models 

## Authors
[Brad Price](https://business.wvu.edu/faculty-and-staff/directory/profile?pid=273); [Maryam Khodaverdi](https://directory.hsc.wvu.edu/Profile/61365); [Brian Hendricks](https://directory.hsc.wvu.edu/Profile/52462); [Gordon S. Smith](https://directory.hsc.wvu.edu/Profile/46172); [Wesley Kimble](https://directory.hsc.wvu.edu/Profile/39623); [Adam Halasz](https://mathanddata.wvu.edu/directory/faculty/adam-halasz); [Sara Guthrie](https://soca.wvu.edu/faculty-and-staff/graduate-student-directory/saraguthrie); [Julia D. Fraustino](https://mediacollege.wvu.edu/faculty-and-staff/profiles/julia-daisy-fraustino); [Sally L. Hodder](https://directory.hsc.wvu.edu/Profile/41751);

## Abstract
Objective:  The goal of this study is to propose and test a scalable framework for machine learning algorithms to predict near-term SARS-CoV-2 cases by incorporating and evaluating the impact of real-time dynamic public health data.

Materials and Methods: Data used in this study include patient-level results, procurement, and location information, of all SARS-CoV-2 tests reported in West Virginia as part of their mandatory reporting system from January 2021-March 2022.  We propose a method for incorporating and comparing widely available public health metrics  inside of a machine learning framework, specifically a Long-Short-Term Memory network, to forecast SARS-CoV-2 cases across various feature sets.

Results:  Our approach provides better prediction of localized case counts and indicates the impact of the dynamic elements of the pandemic on predictions, such as the influence of the mixture of viral variants in the population and variable testing and vaccination rates during various eras of the pandemic.  The decrease in mean absolute percentage error varies between 0.4% and 10% during both the Omicron and Delta period  (depending on the competitor model).  Results also show that models without vaccination information out prerform competitor models in recommending  the top outbreak locations in 8 of 11 weeks studied during the Omicron period.

Discussion: Utilizing real-time public health metrics, including estimated Rt from multiple SARS-CoV-2 variants, vaccination rates, and testing information, provided a significant increase in the accuracy of the model during the Omicron and Delta period, thus providing more precise forecasting of daily case counts at the county level. 
Conclusion: Our proposed framework incorporates available public health metrics with operational data on the impact of testing, vaccination, and current viral variant   mixtures in the population to provide provides a foundation for combining dynamic public health metrics and machine learning models to deliver forecasting and insights in healthcare domains. 
 

## Repository Usage

This repository is broken down into: 

[Code_workbook](https://github.com/MKhodaverdi/COVID19-Enhanced-Case-Prediction-/tree/main/Code_Workbook)

[Data_workbook](https://github.com/MKhodaverdi/COVID19-Enhanced-Case-Prediction-/tree/main/Data_Workbook)


## License
[MIT](https://choosealicense.com/licenses/mit/)
