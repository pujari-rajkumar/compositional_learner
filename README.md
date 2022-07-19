This is the code repository for the paper, Understanding Politics via Contextualized Discourse Processing: https://arxiv.org/abs/2012.15784

Download the required data from here: https://drive.google.com/file/d/1Y-BTF34tnS8FhE0dGViY9MYN7nNrb50N/view?usp=sharing

Extract the zipped archive and place data/ folder in the main directory alongside compositional_reader/, data_processing/ etc,. Make sure that there isn't a nested folder named data/ inside the data/ folder after extraction. Folders inside data/ should be composite_learner_data/, evaluation_data/ and so on.

Follow the steps sequentially to reproduce results from the paper. This repository consists of scripts related to data collection, data processing, learnings tasks and evaluation tasks. If you are only interested in reproducing evaluation results, you don't need to train the entire model again. Trained model parameters are provided in the downloadable data link. Skip forward to evaluation tasks section below.

If you are interested in training compositional reader model on learning tasks as described in the paper, you don't need to run data collection scripts. Data required to train the model, as described in the paper, is provided in the downloadable data link. Skip forward to the data pre-processing section below.

To collect data using collection scripts:
TO DO


To pre-process data for learning tasks:
1. Download staford-corenlp from here: https://stanfordnlp.github.io/corenlp-docs-dev/download.html
2. Extract and place the contents in the main directory in a folder named staford-corenlp/
3. Run the following commands in this order:

        python data_processing/text_processing.py
        python data_processing/document_bert_embedding_computation.py
        python data_processing/query_dict_generation.py

P.S: Don't run them in parallel as the later steps require the output of the earlier steps. Data pre-processing takes a long time (~5 hours)


To train the compositional reader model on learning tasks, change to the main directory with compositional_learner/ and data/ folders and then run:

        python compositional_learner/learning_tasks/authorship_prediction_model.py
        python compositional_learner/learning_tasks/mentioned_entity_prediction_model.py
	
Upon running the two commands sequentially, the trained_models will be saved in ./data/composite_learner_data/saved_parameters/ folder. You may use the saved paramters to initialize the parameters for further tasks. Alternatively, you may use the parameters provided in the data download link above for inference.



To run evaluation tasks:

1. To generate visualizations for 'politicians on all issues' (Fig. 5 and Appendix Fig.s 4-13) and 'comaprison of politicians stances on issues' (Fig. 4), you may run:

                                    python ./evaluation_tasks/visualizations.py

	It runs on CPU and will take 2-5 secs to finish


2. To generate results of 'NRA Grades Paraphrase Task' (Tab. 4), 'NRA Grade Prediction Task' (Tab 5., Fig. 4 and Appendix Fig. 2), LCV Score Prediction Task (Appendix Tab. 1, Fig.s 1 & 2), you may run:

                                    python ./evaluation_tasks/grade_prediction.py


	This also runs on CPU and may take up to 30 minutes to finish. It is trains and evaluates a GradePredictor feed-forward neural network 320 times (5 random seeds * 8 training data sizes * 4 models * 2 tasks). We didn't provide the option of GPU because it doesn't take too much time to finish.


3. Both the scripts will generate plots as .png  images in the ./data/evaluation_data/ folder. grade_prediction.py generates training output in ./data/evaluation_data/grade_pred_log.txt.

4. We provided representations generated by all models for all entities for all issues in the ./data/evaluation_data/ folder. They are named entity_issue_<model_name>.pkl.
