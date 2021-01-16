This is the code repository for the paper, Understanding Politics via Contextualized Discourse Processing: https://arxiv.org/abs/2012.15784

The repository is only partially full yet. The first version will be ready by 01/20/2021. Sorry for the delay.

Download the required data from here: https://drive.google.com/file/d/1Y-BTF34tnS8FhE0dGViY9MYN7nNrb50N/view?usp=sharing
Place the contents in a folder named data/ in the main repository. Make sure that there isn't a nested folder named data/ inside the data/ folder after extraction. Folders inside data/ should be composite_learner_data/, evaluation_data/, etc,.

Follow the steps sequentially to reproduce results from the paper. This repository consists of scripts related to data collection, data processing, learnings tasks and evaluation tasks. If you are only interested in reproducing evaluation results, you don't need to train the entire model again as trained model parameters are provided in the downloadable data link. Skip forward to evaluation tasks section below.

If you are interested in training compositional reader model on learning tasks as described in the paper, you don't need to run data collection scripts. Data required to train the model as described in the paper is provided in the link above. Start from the data pre-processing step below.

To collect data using collection scripts:
TO DO


To pre-process data for learning tasks:
TO DO


To train the compositional reader model on learning tasks:
TO DO


To run evaluation tasks:
TO DO


1. To generate visualizations for 'politicians on all issues' (Fig. 5 and Appendix Fig.s 4-13) and 'comaprison of politicians stances on issues' (Fig. 4), you may run:

                                    python3.6 ./evaluation_tasks/visualizations.py

	It runs on CPU and will take 2-5 secs to finish


2. To generate results of 'NRA Grades Paraphrase Task' (Tab. 4), 'NRA Grade Prediction Task' (Tab 5., Fig. 4 and Appendix Fig. 2), LCV Score Prediction Task (Appendix Tab. 1, Fig.s 1 & 2), you may run:

                                    python3.6 ./evaluation_tasks/grade_prediction.py


	This also runs on CPU and may take up to 30 minutes to finish. It is trains and evaluates a GradePredictor feed-forward neural network 320 times (5 random seeds * 8 training data sizes * 4 models * 2 tasks). We didn't provide the option of GPU because it doesn't take too much time to finish.


3. Both the scripts will generate plots as .png  images in the ./data/evaluation_data/ folder. grade_prediction.py generates training output in ./data/evaluation_data/grade_pred_log.txt.

4. We provided representations generated by all models for all entities for all issues in the ./data/evaluation_data/ folder. They are named entity_issue_<model_name>.pkl.
