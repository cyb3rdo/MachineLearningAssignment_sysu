## **Intro**
本项目为**机器学习**课程的大作业，主要任务是基于[ReChorus](https://github.com/THUwangcy/ReChorus)框架，对论文[Denoising Implicit Feedback for Recommendation.](https://arxiv.org/abs/2006.04153)中的ADT(Adaptive Denoising Training)训练策略尝试复现。

## **代码修改**
- **增加的数据集**：位于`./ReChorus/data/MIND_Small`，由于文件大小原因，以zip格式上传，运行前请解压。
- **增加的Runner**：位于`./ReChorus/src/helpers/ADTRunner.py`
- **增加的模型代码** ：位于`./ReChorus/src/models/context/FM_ADT.py`以及`./ReChorus/src/models/context/WideDeep_ADT.py`

## **实验部分**
```bash
cd ./ReChorus/src
python main.py --model_name FM --lr 1e-3 --l2 1e-4 --dataset ML_1MCTR --path ../data/MovieLens_1M --num_neg 0 --batch_size 1024 --metric AUC,F1_SCORE --include_item_features 1 --include_situation_features 1 --model_mode CTR --loss_n BCE
python main.py --model_name FM_ADT --lr 1e-3 --l2 1e-4 --dataset ML_1MCTR --path ../data/MovieLens_1M --num_neg 0 --batch_size 1024 --metric AUC,F1_SCORE --include_item_features 1 --include_situation_features 1 --model_mode CTR --loss_n BCE --paradigm R_CE --beta 0.2
python main.py --model_name FM_ADT --lr 1e-3 --l2 1e-4 --dataset ML_1MCTR --path ../data/MovieLens_1M --num_neg 0 --batch_size 1024 --metric AUC,F1_SCORE --include_item_features 1 --include_situation_features 1 --model_mode CTR --loss_n BCE --paradigm T_CE --drop_rate 0.2 --num_gradual 30000
python main.py --model_name WideDeep --lr 5e-3 --l2 0 --dropout 0.5 --layers "[64,64,64]" --dataset ML_1MCTR --path ../data/MovieLens_1M --num_neg 0 --batch_size 1024 --metric AUC,F1_SCORE --include_item_features 1 --include_situation_features 1 --model_mode CTR --loss_n BCE
python main.py --model_name WideDeep_ADT --lr 5e-3 --l2 0 --dropout 0.5 --layers "[64,64,64]" --dataset ML_1MCTR --path ../data/MovieLens_1M --num_neg 0 --batch_size 1024 --metric AUC,F1_SCORE --include_item_features 1 --include_situation_features 1 --model_mode CTR --loss_n BCE --paradigm R_CE
python main.py --model_name WideDeep_ADT --lr 5e-3 --l2 0 --dropout 0.5 --layers "[64,64,64]" --dataset ML_1MCTR --path ../data/MovieLens_1M --num_neg 0 --batch_size 1024 --metric AUC,F1_SCORE --include_item_features 1 --include_situation_features 1 --model_mode CTR --loss_n BCE --paradigm T_CE --num_gradual 30000
python main.py --model_name FM --lr 5e-4 --l2 0 --dataset MINDCTR --path ../data/MIND_Small --num_neg 0 --batch_size 1024 --metric AUC --include_item_features 1 --include_situation_features 1 --model_mode CTR --loss_n BCE
python main.py --model_name FM_ADT --lr 5e-4 --l2 0 --dataset MINDCTR --path ../data/MIND_Small --num_neg 0 --batch_size 1024 --metric AUC --include_item_features 1 --include_situation_features 1 --model_mode CTR --loss_n BCE --paradigm R_CE
python main.py --model_name FM_ADT --lr 5e-4 --l2 0 --dataset MINDCTR --path ../data/MIND_Small --num_neg 0 --batch_size 1024 --metric AUC --include_item_features 1 --include_situation_features 1 --model_mode CTR --loss_n BCE --paradigm T_CE --drop_rate 0.1 --num_gradual 60000
python main.py --model_name WideDeep --lr 5e-3 --l2 0 --dropout 0.5 --layers "[64,64,64]" --dataset MINDCTR --path ../data/MIND_Small --num_neg 0 --batch_size 1024 --metric AUC --include_item_features 1 --include_situation_features 1 --model_mode CTR --loss_n BCE
python main.py --model_name WideDeep_ADT --lr 5e-3 --l2 0 --dropout 0.5 --layers "[64,64,64]" --dataset MINDCTR --path ../data/MIND_Small --num_neg 0 --batch_size 1024 --metric AUC --include_item_features 1 --include_situation_features 1 --model_mode CTR --loss_n BCE --paradigm R_CE
python main.py --model_name WideDeep_ADT --lr 5e-3 --l2 0 --dropout 0.5 --layers "[64,64,64]" --dataset MINDCTR --path ../data/MIND_Small --num_neg 0 --batch_size 1024 --metric AUC --include_item_features 1 --include_situation_features 1 --model_mode CTR --loss_n BCE --paradigm T_CE -drop_rate 0.1 --num_gradual 30000
```

## **Citation**
```
@article{li2024rechorus2,
  title={ReChorus2. 0: A Modular and Task-Flexible Recommendation Library},
  author={Li, Jiayu and Li, Hanyu and He, Zhiyu and Ma, Weizhi and Sun, Peijie and Zhang, Min and Ma, Shaoping},
  journal={arXiv preprint arXiv:2405.18058},
  year={2024}
}
```
```
@inproceedings{wang2021denoising,
  title={Denoising implicit feedback for recommendation},
  author={Wang, Wenjie and Feng, Fuli and He, Xiangnan and Nie, Liqiang and Chua, Tat-Seng},
  booktitle={Proceedings of the 14th ACM international conference on web search and data mining},
  pages={373--381},
  publisher={ACM},
  year={2021}
}
```
