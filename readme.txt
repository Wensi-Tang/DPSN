Environment
python = 3.5
sklearn = 0.22.2.post1 
pyts = 0.7.0


Full result
	.\results\result.txt
visulization result of full result 
	.\time_series_proto\image_result\!A.pdf
Full interpretability result
	.\SFA_Python-master\test\image_result


How to run

	Get basline and feature extraction

		go to folder   "~/DPSN/SFA_Python-master/test"
		using jupyter notebook run from 1_* to 4_* which are:

			1_get_SFA_hyper_parameter.ipynb
			2_split_dataset_and_prepare_SFA_feature_for_small_dataset.ipynb
			3_get_ST_hyper_parameter.ipynb
			4_get_ST_result.ipynb
        
	DPSN Classification    
		go to folder   "~/DPSN/time_series_proto"
		using terminal to run
			bash scripts/train_proto.sh 0 2 linear_transform ./exp/train.py
			## if you want more dataset change that "name_list" in ./exp/train.py
    
	DPSN inter  
		go to folder  "~/DPSN/time_series_proto/exp/"
		using terminal to run
			bash test_load-shapelet.sh
			## if you want more dataset change that "name_list" in ./exp/test_load_model_shapelet.py
    
		go to folder "~/DPSN/SFA_Python-master/test"
		using jupyter notebook to run 
			5_interpretable_shapelet.ipynb




	