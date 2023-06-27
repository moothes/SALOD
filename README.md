
# SALOD
Here is the SALient Object Detection (SALOD) benchmark ([paper link](https://arxiv.org/abs/2202.02925)).  

We have re-implemented over 20 SOD methods using the same settings, including input size, data loader and evaluation metrics (thanks to [Metrics](https://github.com/lartpang/Py-SOD-VOS-EvalToolkit)). Some other networks are debugging now, it is welcome for your contributions on these models.

*Our new unsupervised SOD method [A2S-v2 framework](https://github.com/moothes/A2S-v2) was accepted by CVPR 2023!*

**You can contact me through the official email: zhouhj26@mail2.sysu.edu.cn**  

# Datasets
Our SALOD dataset can be downloaded from: [SALOD](https://drive.google.com/file/d/1kxhUoWUAnFhOE_ZoA1www8msG2pKHg3_/view?usp=sharing).   
Original SOD datasets from: [SOD](https://drive.google.com/file/d/17X4SiSVuBmqkvQJe_ScVARKPM_vgvCOi/view?usp=sharing), including DUTS-TR,DUTS-TE,ECSSD,SOD,PASCAL-S,HKU-IS,DUT-OMRON.  
COD datasets from: [COD](https://drive.google.com/file/d/1zUgaGxr9PeDcfLfBisV2q8QXL6Tp1QzC/view?usp=sharing), including COD-TR (COD-TR + CAMO-TR), COD-TE, CAMO-TE, NC4K.
 
# Results
All models are trained with the following setting:  
1. ```--strategy=sche_f3net``` for the latest training strategy as original F3Net, LDF, PFSNet and CTDNet;
2. ```--multi``` for multi-scale training;
3. ```--data_aug``` for random croping;  
4. 1 * BCE_loss + 1 * IOU_loss as loss.

## Benchmarking results
Following the above settings, we list the benchmark results here.
All weights can be downloaded from [Baidu disk](https://pan.baidu.com/s/1KXFU09nBElHqP9ffdHWtNw) [pqn6].  
**Noted that FPS is tested on our device with batch_size=1, you should test all methods and report the scores on your own device.**

Methods | #Para. | GMACs  | FPS  | max-F | ave-F | Fbw  | MAE  | SM   | em    
 ----   | ---    | -----  | ---- | ----- | ----- | ---- | ---- | ---- | -----  
DHSNet  | 24.2   | 13.8   | 49.2 | .909  | .871  | .863 | .037 | .905 | .925      
Amulet  | 79.8   | 1093.8 | 35.1 | .897  | .856  | .846 | .042 | .896 | .919   
NLDF    | 41.1   | 115.1  | 30.5 | .908  | .868  | .859 | .038 | .903 | .930  
SRM     | 61.2   | 20.2   | 34.3 | .893  | .851  | .841 | .042 | .892 | .925    
DSS     | 134.3  | 35.3   | 27.3 | .906  | .868  | .859 | .038 | .901 | .933 
PiCaNet | 106.1  | 36.9   | 14.8 | .900  | .864  | .852 | .043 | .896 | .924  
BASNet  | 95.5   | 47.2   | 32.8 | .911  | .872  | .863 | .040 | .905 | .925  
CPD     | 47.9   | 14.7   | 22.7 | .913  | .884  | .874 | .034 | .911 | .938 
PoolNet | 68.3   | 66.9   | 33.9 | .916  | .882  | .875 | .035 | .911 | .938  
EGNet   | 111.7  | 222.8  | 10.2 | .913  | .884  | .875 | .036 | .908 | .936  
SCRN    | 25.2   | 12.5   | 19.3 | .916  | .881  | .872 | .035 | .910 | .935 
F3Net   | 25.5   | 13.6   | 39.2 | .911  | .878  | .869 | .036 | .908 | .932  
GCPA    | 67.1   | 54.3   | 37.8 | .914  | .884  | .874 | .036 | .910 | .937   
ITSD    | 25.7   | 19.6   | 29.4 | .918  | .880  | .873 | .037 | .910 | .932   
MINet   | 162.4  | 87     | 23.5 | .912  | .874  | .866 | .038 | .908 | .931 
LDF     | 25.2   | 12.8   | 37.5 | .913  | .879  | .873 | .035 | .909 | .938  
GateNet | 128.6  | 96     | 25.9 | .912  | .882  | .870 | .037 | .906 | .934 
PFSNet  | 31.2   | 37.5   | 21.7 | .912  | .879  | .865 | .038 | .904 | .931 
CTDNet  | 24.6   | 10.2   | 64.2 | .918  | .887  | .880 | .033 | .913 | .940  
EDN     | 35.1   | 16.1   | 27.4 | .916  | .883  | .875 | .036 | .910 | .934  


## Conventional SOD results
The orig. means the results of official saliency predictions, while ours are the re-implemented results in our benchmark.
The weights of these models can be downloaded from: [Baidu Disk](https://pan.baidu.com/s/1ByHuao32_2fUSXV7nNNMIA)(cs6u)    

<table>
 <tr>
<td width=60 rowspan=2>Method</td>
<td width=60 rowspan=2>Src</td>
<td width=100 colspan=2>PASCAL-S</td>
<td width=100 colspan=2>ECSSD</td>
<td width=100 colspan=2>HKU-IS</td>
<td width=100 colspan=2>DUTS-TE</td>
<td width=100 colspan=2>DUT-OMRON</td>
 </tr>
 <tr>
<td>max-F</td><td>MAE</td>
<td>max-F</td><td>MAE</td>
<td>max-F</td><td>MAE</td>
<td>max-F</td><td>MAE</td>
<td>max-F</td><td>MAE</td>
 </tr>
 <tr>
<td rowspan=2>DHSNet</td>
<td>orig.</td>
<td>.820</td><td>.091</td>
<td>.906</td><td>.059</td>
<td>.890</td><td>.053</td>
<td>.808</td><td>.067</td>
<td>--</td><td>--</td>
 </tr>
 <tr>
<td>ours</td>
<td>.870</td><td>.063</td>
<td>.944</td><td>.036</td>
<td>.935</td><td>.031</td>
<td>.887</td><td>.040</td>
<td>.805</td><td>.062</td>
 </tr>
 <tr>
<td rowspan=2>Amulet</td>
<td>orig.</td>
<td>.828</td><td>.100</td>
<td>.915</td><td>.059</td>
<td>.897</td><td>.051</td>
<td>.778</td><td>.085</td>
<td>.743</td><td>.098</td>
 </tr>
 <tr>
<td>ours</td>
<td>.871</td><td>.066</td>
<td>.936</td><td>.045</td>
<td>.928</td><td>.036</td>
<td>.871</td><td>.044</td>
<td>.791</td><td>.065</td>
 </tr>
 <tr>
<td rowspan=2>NLDF</td>
<td>orig.</td>
<td>.822</td><td>.098</td>
<td>.905</td><td>.063</td>
<td>.902</td><td>.048</td>
<td>.813</td><td>.065</td>
<td>.753</td><td>.080</td>
 </tr>
 <tr>
<td>ours</td>
<td>.872</td><td>.064</td>
<td>.937</td><td>.042</td>
<td>.927</td><td>.035</td>
<td>.882</td><td>.044</td>
<td>.796</td><td>.068</td>
 </tr>
 <tr>
<td rowspan=2>SRM</td>
<td>orig.</td>
<td>.838</td><td>.084</td>
<td>.917</td><td>.054</td>
<td>.906</td><td>.046</td>
<td>.826</td><td>.059</td>
<td>.769</td><td>.069</td>
 </tr>
 <tr>
<td>ours</td>
<td>.854</td><td>.069</td>
<td>.922</td><td>.046</td>
<td>.904</td><td>.043</td>
<td>.846</td><td>.049</td>
<td>.774</td><td>.068</td>
 </tr>
 <tr>
<td rowspan=2>DSS</td>
<td>orig.</td>
<td>.831</td><td>.093</td>
<td>.921</td><td>.052</td>
<td>.900</td><td>.050</td>
<td>.826</td><td>.065</td>
<td>.769</td><td>.063</td>
 </tr>
 <tr>
<td>ours</td>
<td>.870</td><td>.063</td>
<td>.937</td><td>.039</td>
<td>.924</td><td>.035</td>
<td>.878</td><td>.040</td>
<td>.800</td><td>.059</td>
 </tr>
 <tr>
<td rowspan=2>PiCANet</td>
<td>orig.</td>
<td>.857</td><td>.076</td>
<td>.935</td><td>.046</td>
<td>.918</td><td>.043</td>
<td>.860</td><td>.051</td>
<td>.803</td><td>.065</td>
 </tr>
 <tr>
<td>ours</td>
<td>.867</td><td>.074</td>
<td>.938</td><td>.044</td>
<td>.927</td><td>.036</td>
<td>.879</td><td>.046</td>
<td>.798</td><td>.077</td>
 </tr>
 <tr>
<td rowspan=2>BASNet</td>
<td>orig.</td>
<td>.854</td><td>.076</td>
<td>.942</td><td>.037</td>
<td>.928</td><td>.032</td>
<td>.859</td><td>.048</td>
<td>.805</td><td>.056</td>
 </tr>
 <tr>
<td>ours</td>
<td>.884</td><td>.057</td>
<td>.950</td><td>.034</td>
<td>.943</td><td>.028</td>
<td>.907</td><td>.033</td>
<td>.833</td><td>.052</td>
 </tr>
 <tr>
<td rowspan=2>CPD</td>
<td>orig.</td>
<td>.859</td><td>.071</td>
<td>.939</td><td>.037</td>
<td>.925</td><td>.034</td>
<td>.865</td><td>.043</td>
<td>.797</td><td>.056</td>
 </tr>
 <tr>
<td>ours</td>
<td>.883</td><td>.057</td>
<td>.946</td><td>.034</td>
<td>.934</td><td>.031</td>
<td>.892</td><td>.037</td>
<td>.815</td><td>.059</td>
 </tr>
 <tr>
<td rowspan=2>PoolNet</td>
<td>orig.</td>
<td>.863</td><td>.075</td>
<td>.944</td><td>.039</td>
<td>.931</td><td>.034</td>
<td>.880</td><td>.040</td>
<td>.808</td><td>.056</td>
 </tr>
 <tr>
<td>ours</td>
<td>.877</td><td>.062</td>
<td>.946</td><td>.035</td>
<td>.936</td><td>.030</td>
<td>.895</td><td>.037</td>
<td>.812</td><td>.063</td>
 </tr>
 <tr>
<td rowspan=2>EGNet</td>
<td>orig.</td>
<td>.865</td><td>.074</td>
<td>.947</td><td>.037</td>
<td>.934</td><td>.032</td>
<td>.889</td><td>.039</td>
<td>.815</td><td>.053</td>
 </tr>
 <tr>
<td>ours</td>
<td>.880</td><td>.060</td>
<td>.948</td><td>.032</td>
<td>.937</td><td>.030</td>
<td>.892</td><td>.037</td>
<td>.812</td><td>.058</td>
 </tr>
 <tr>
<td rowspan=2>SCRN</td>
<td>orig.</td>
<td>.877</td><td>.063</td>
<td>.950</td><td>.037</td>
<td>.934</td><td>.034</td>
<td>.888</td><td>.040</td>
<td>.811</td><td>.056</td>
 </tr>
 <tr>
<td>ours</td>
<td>.871</td><td>.063</td>
<td>.947</td><td>.037</td>
<td>.934</td><td>.032</td>
<td>.895</td><td>.039</td>
<td>.813</td><td>.063</td>
 </tr>
 <tr>
<td rowspan=2>F3Net</td>
<td>orig.</td>
<td>.872</td><td>.061</td>
<td>.945</td><td>.033</td>
<td>.937</td><td>.028</td>
<td>.891</td><td>.035</td>
<td>.813</td><td>.053</td>
 </tr>
 <tr>
<td>ours</td>
<td>.884</td><td>.057</td>
<td>.950</td><td>.033</td>
<td>.937</td><td>.030</td>
<td>.903</td><td>.034</td>
<td>.819</td><td>.053</td>
 </tr>
 <tr>
<td rowspan=2>GCPA</td>
<td>orig.</td>
<td>.869</td><td>.062</td>
<td>.948</td><td>.035</td>
<td>.938</td><td>.031</td>
<td>.888</td><td>.038</td>
<td>.812</td><td>.056</td>
 </tr>
 <tr>
<td>ours</td>
<td>.885</td><td>.056</td>
<td>.951</td><td>.031</td>
<td>.941</td><td>.028</td>
<td>.905</td><td>.034</td>
<td>.820</td><td>.055</td>
 </tr>
 <tr>
<td rowspan=2>ITSD</td>
<td>orig.</td>
<td>.872</td><td>.065</td>
<td>.946</td><td>.035</td>
<td>.935</td><td>.030</td>
<td>.885</td><td>.040</td>
<td>.821</td><td>.059</td>
 </tr>
 <tr>
<td>ours</td>
<td>.880</td><td>.067</td>
<td>.950</td><td>.036</td>
<td>.939</td><td>.030</td>
<td>.895</td><td>.040</td>
<td>.817</td><td>.072</td>
 </tr>
 <tr>
<td rowspan=2>MINet</td>
<td>orig.</td>
<td>.867</td><td>.064</td>
<td>.947</td><td>.033</td>
<td>.935</td><td>.029</td>
<td>.884</td><td>.037</td>
<td>.810</td><td>.056</td>
 </tr>
 <tr>
<td>ours</td>
<td>.874</td><td>.064</td>
<td>.947</td><td>.036</td>
<td>.937</td><td>.031</td>
<td>.893</td><td>.039</td>
<td>.816</td><td>.061</td>
 </tr>
 <tr>
<td rowspan=2>LDF</td>
<td>orig.</td>
<td>.874</td><td>.060</td>
<td>.950</td><td>.034</td>
<td>.939</td><td>.028</td>
<td>.898</td><td>.034</td>
<td>.820</td><td>.052</td>
 </tr>
 <tr>
<td>ours</td>
<td>.883</td><td>.058</td>
<td>.951</td><td>.032</td>
<td>.940</td><td>.029</td>
<td>.903</td><td>.035</td>
<td>.818</td><td>.058</td>
 </tr>
 <tr>
<td rowspan=2>GateNet</td>
<td>orig.</td>
<td>.869</td><td>.067</td>
<td>.945</td><td>.040</td>
<td>.933</td><td>.033</td>
<td>.888</td><td>.040</td>
<td>.818</td><td>.055</td>
 </tr>
 <tr>
<td>ours</td>
<td>.867</td><td>.066</td>
<td>.944</td><td>.037</td>
<td>.934</td><td>.031</td>
<td>.891</td><td>.039</td>
<td>.803</td><td>.062</td>
 </tr>
 <tr>
<td rowspan=2>PFSNet</td>
<td>orig.</td>
<td>.875</td><td>.063</td>
<td>.952</td><td>.031</td>
<td>.943</td><td>.026</td>
<td>.896</td><td>.036</td>
<td>.823</td><td>.055</td>
 </tr>
 <tr>
<td>ours</td>
<td>.883</td><td>.060</td>
<td>.950</td><td>.034</td>
<td>.939</td><td>.030</td>
<td>.899</td><td>.037</td>
<td>.816</td><td>.063</td>
 </tr>
 <tr>
<td rowspan=2>CTDNet</td>
<td>orig.</td>
<td>.878</td><td>.061</td>
<td>.950</td><td>.032</td>
<td>.941</td><td>.027</td>
<td>.897</td><td>.034</td>
<td>.826</td><td>.052</td>
 </tr>
 <tr>
<td>ours</td>
<td>.885</td><td>.057</td>
<td>.950</td><td>.031</td>
<td>.940</td><td>.028</td>
<td>.904</td><td>.033</td>
<td>.821</td><td>.055</td>
 </tr>
 <tr>
<td rowspan=2>EDN</td>
<td>orig.</td>
<td>.880</td><td>.062</td>
<td>.951</td><td>.032</td>
<td>.941</td><td>.026</td>
<td>.895</td><td>.035</td>
<td>.828</td><td>.049</td>
 </tr>
 <tr>
<td>ours</td>
<td>.891</td><td>.058</td>
<td>.953</td><td>.031</td>
<td>.945</td><td>.027</td>
<td>.910</td><td>.032</td>
<td>.837</td><td>.055</td>
 </tr>
</table>


# Available Methods:

 Methods | Publish. | Paper | Src Code
 ----    | -----    | ----- | ------ 
 MENet | CVPR 2023 | [openaccess](https://openaccess.thecvf.com/content/CVPR2023/html/Wang_Pixels_Regions_and_Objects_Multiple_Enhancement_for_Salient_Object_Detection_CVPR_2023_paper.html) | [PyTorch](https://github.com/yiwangtz/MENet)
 EDN  | TIP 2022 | [TIP](https://ieeexplore.ieee.org/abstract/document/9756227/)|  [Pytorch](https://github.com/yuhuan-wu/EDN)
 CTDNet  | ACM MM 2021 | [ACM](https://dl.acm.org/doi/abs/10.1145/3474085.3475494?casa_token=eKn8q7l2hJEAAAAA%3A4YGBXBpC6cCcFdpekxbaZncgBEru_mi69kNixfZSPeFRhD2gkeKpXIZyuiIW1bH80IuNV9ANmBw)|  [Pytorch](https://github.com/zhaozhirui/CTDNet)
 PFSNet  | AAAI 2021 | [AAAI.org](https://ojs.aaai.org/index.php/AAAI/article/view/16331)|  [Pytorch](https://github.com/iCVTEAM/PFSNet)
 GateNet | ECCV 2020 | [springer](https://link.springer.com/chapter/10.1007/978-3-030-58536-5_3)|  [Pytorch](https://github.com/Xiaoqi-Zhao-DLUT/GateNet-RGB-Saliency)
 LDF     | CVPR 2020 | [openaccess](https://openaccess.thecvf.com/content_CVPR_2020/html/Wei_Label_Decoupling_Framework_for_Salient_Object_Detection_CVPR_2020_paper.html)|  [Pytorch](https://github.com/weijun88/LDF)
 MINet   | CVPR 2020 | [openaccess](https://openaccess.thecvf.com/content_CVPR_2020/papers/Pang_Multi-Scale_Interactive_Network_for_Salient_Object_Detection_CVPR_2020_paper.pdf) | [Pytorch](https://github.com/lartpang/MINet)  
 ITSD    | CVPR 2020 | [openaccess](https://openaccess.thecvf.com/content_CVPR_2020/papers/Zhou_Interactive_Two-Stream_Decoder_for_Accurate_and_Fast_Saliency_Detection_CVPR_2020_paper.pdf) | [Pytorch](https://github.com/moothes/ITSD-pytorch)  
 GCPA    | AAAI 2020 | [aaai.org](https://aaai.org/ojs/index.php/AAAI/article/view/6633) | [Pytorch](https://github.com/JosephChenHub/GCPANet)  
 F3Net   | AAAI 2020 | [aaai.org](https://aaai.org/ojs/index.php/AAAI/article/view/6916) | [Pytorch](https://github.com/weijun88/F3Net)  
 SCRN    | ICCV 2019 | [openaccess](https://openaccess.thecvf.com/content_ICCV_2019/papers/Wu_Stacked_Cross_Refinement_Network_for_Edge-Aware_Salient_Object_Detection_ICCV_2019_paper.pdf) | [Pytorch](https://github.com/wuzhe71/SCRN)  
 EGNet   | ICCV 2019 | [openaccess](https://openaccess.thecvf.com/content_ICCV_2019/papers/Zhao_EGNet_Edge_Guidance_Network_for_Salient_Object_Detection_ICCV_2019_paper.pdf) | [Pytorch](https://github.com/JXingZhao/EGNet)  
 PoolNet | CVPR 2019 | [openaccess](https://openaccess.thecvf.com/content_CVPR_2019/papers/Liu_A_Simple_Pooling-Based_Design_for_Real-Time_Salient_Object_Detection_CVPR_2019_paper.pdf) | [Pytorch](https://github.com/backseason/PoolNet)  
 CPD     | CVPR 2019 | [openaccess](https://openaccess.thecvf.com/content_CVPR_2019/papers/Wu_Cascaded_Partial_Decoder_for_Fast_and_Accurate_Salient_Object_Detection_CVPR_2019_paper.pdf) | [Pytorch](https://github.com/wuzhe71/CPD)  
 BASNet  | CVPR 2019 | [openaccess](https://openaccess.thecvf.com/content_CVPR_2019/papers/Qin_BASNet_Boundary-Aware_Salient_Object_Detection_CVPR_2019_paper.pdf) | [Pytorch](https://github.com/NathanUA/BASNet)  
 DSS     | TPAMI 2019| [IEEE](https://ieeexplore.ieee.org/document/8315520/)/[ArXiv](https://arxiv.org/abs/1611.04849) | [Pytorch](https://github.com/AceCoooool/DSS-pytorch)  
 PicaNet | CVPR 2018 | [openaccess](https://openaccess.thecvf.com/content_cvpr_2018/papers/Liu_PiCANet_Learning_Pixel-Wise_CVPR_2018_paper.pdf) | [Pytorch](https://github.com/Ugness/PiCANet-Implementation)  
 SRM     | ICCV 2017 | [openaccess](https://openaccess.thecvf.com/content_ICCV_2017/papers/Wang_A_Stagewise_Refinement_ICCV_2017_paper.pdf) | [Pytorch](https://github.com/xsxszab/SRM-Pytorch) 
 Amulet  | ICCV 2017 | [openaccess](https://openaccess.thecvf.com/content_ICCV_2017/papers/Zhang_Amulet_Aggregating_Multi-Level_ICCV_2017_paper.pdf) | [Pytorch](https://github.com/xsxszab/Amulet-Pytorch)  
 NLDF    | CVPR 2017 | [openaccess](https://openaccess.thecvf.com/content_cvpr_2017/papers/Luo_Non-Local_Deep_Features_CVPR_2017_paper.pdf) | [Pytorch](https://github.com/AceCoooool/NLDF-pytorch)/[TF](https://github.com/zhimingluo/NLDF) 
 DHSNet  | CVPR 2016 | [openaccess](https://openaccess.thecvf.com/content_cvpr_2016/papers/Liu_DHSNet_Deep_Hierarchical_CVPR_2016_paper.pdf) | [Pytorch](https://github.com/xsxszab/DHSNet-Pytorch)  
 `Tuning`  | -----   | ----- | -----
 *PAGE    | CVPR2019 | [openaccess](https://openaccess.thecvf.com/content_CVPR_2019/papers/Wang_Salient_Object_Detection_With_Pyramid_Attention_and_Salient_Edges_CVPR_2019_paper.pdf) | [TF](https://github.com/wenguanwang/PAGE-Net)  
 *PFA     | CVPR2019 | [openaccess](https://openaccess.thecvf.com/content_CVPR_2019/papers/Zhao_Pyramid_Feature_Attention_Network_for_Saliency_Detection_CVPR_2019_paper.pdf) | [Pytorch](https://github.com/dizaiyoufang/pytorch_PFAN) 
 *PFPN   | AAAI2020 | [aaai.org](https://ojs.aaai.org/index.php/AAAI/article/view/6892) | [Pytorch](https://github.com/chenquan-cq/PFPN)
 
 

 # Usage
 
 ```
 # model_name: lower-cased method name. E.g. poolnet, egnet, gcpa, dhsnet or minet.
 python3 train.py model_name --gpus=0 --trset=[DUTS-TR,SALOD,COD-TR]
 
 python3 test.py model_name --gpus=0 --weight=path_to_weight [--save]
 
 python3 test_fps.py model_name --gpus=0
 
 # To evaluate generated maps:
 python3 eval.py --pre_path=path_to_maps
 ```
 
# Loss Factory

We supply a **Loss Factory** for an easier way to tune the loss functions.
You can set --loss and --lw parameters to use it.

Here are some examples:
```
loss_dict = {'b': BCE, 's': SSIM, 'i': IOU, 'd': DICE, 'e': Edge, 'c': CTLoss}

python train.py basnet --loss=bd
# loss = 1 * bce_loss + 1 * dice_loss

python train.py basnet --loss=bs --lw=0.3,0.7
# loss = 0.3 * bce_loss + 0.7 * ssim_loss

python train.py basnet --loss=bsid --lw=0.3,0.1,0.5,0.2
# loss = 0.3 * bce_loss + 0.1 * ssim_loss + 0.5 * iou_loss + 0.2 * dice_loss
```

# Update Log
2023/06/27:  
* MENet (CVPR 2023) is available, but need more time for achiveving SOTA performance.
*  
2023/03/17:  
* Re-organize the structure of our code. 

2022/12/07:  
* Update conventional SOD results and weights.

2022/10/17:  
* Use ```timm``` library for more backbones.
* Code update.
* Benchmark results update.

2022/08/09:  
* Remove loss.py for each method. The loss functions are defined in config.py now.  
* Weights are uploaded to Baidu Disk.

2022/06/14: 
* New model: EDN (TIP 2022).

2022/05/25:    
* In the previous versions, we found that images with large salient regions get 0 ave-F scores, and thus we obtain lower ave-F scores than their original paper. Now, we fix this bug by adding a round function before evaluating.

2022/05/15: 
* New models: F3Net (AAAI 2020), LDF (CVPR 2020), GateNet (ECCV 2020), PFSNet (AAAI 20221), CTDNet (ACM MM 2021). More models for SOD and COD tasks are coming soon. 
* New dataset: training on COD task is available now.
* Training strategy update. We notice that training strategy is very important for achieving SOTA performance. A new strategy factory is added to /base/strategy.py.

Thanks for citing our work
```xml
@article{salod,
  title={Benchmarking Deep Models for Salient Object Detection},
  author={Zhou, Huajun and Lin, Yang and Yang, Lingxiao and Lai, Jianhuang and Xie, Xiaohua},
  journal={arXiv preprint arXiv:2202.02925},
  year={2022}
}
```
