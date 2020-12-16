# Reproduction

We only report max-F score here. You can [download] the trained models or official maps to get more results. 
 
Src indicates results from the saliency maps that provided by author or generated from source code, while ss and ms mean single-scale and multi-scale training strategies respectively.
 
**Notice: please contact us if you get better results.**
 
 
 Methods | Source | Backbone  | SOD   | Pascal-S | ECSSD | HKU-IS | DUTS-TE | DUT-OMRON 
  ----   | ---    | -----     | ----- | -------- | ----- | -----  | -----   | -----     
 DHSNet  | src    | VGG16     | .827  | .820     | .906  | .890   | .808    | ------    
 ----    | ss     | Resnet50  | .868  | .865     | .940  | .930   | .870    | .796      
 ----    | ms     | Resnet50  | .873  | .864     | .943  | .930   | .880    | .807      
 Amulet  | src    | VGG16     | .798  | .828     | .915  | .897   | .778    | .743      
 ----    | ss     | Resnet50  | .868  | .856     | .933  | .924   | .860    | .781      
 ----    | ms     | Resnet50  | .861  | .861     | .933  | .925   | .867    | .788      
 NLDF    | src    | VGG16     | .841  | .822     | .905  | .902   | .813    | .753      
 ----    | ss     | Resnet50  | .866  | .861     | .933  | .923   | .867    | .792      
 ----    | ms     | Resnet50  | .865  | .869     | .934  | .924   | .873    | .795      
 SRM     | src    | Resnet50  | .843  | .838     | .917  | .906   | .826    | .769      
 ----    | ss     | Resnet50  | .841  | .835     | .919  | .901   | .824    | .763      
 ----    | ms     | Resnet50  | .847  | .840     | .922  | .904   | .832    | .773      
 PicaNet | src    | Resnet50  | .856  | .857     | .935  | .918   | .860    | .803      
 ----    | ss     | Resnet50  | .867  | .857     | .934  | .923   | .869    | .797      
 ----    | ms     | Resnet50  | .860  | .857     | .933  | .917   | .867    | .800      
 DSS     | src    | Resnet50  | .846  | .831     | .921  | .900   | .826    | .769      
 ----    | ss     | Resnet50  | .857  | .852     | .929  | .916   | .855    | .784      
 ----    | ms     | Resnet50  | .858  | .853     | .927  | .914   | .858    | .785      
 BasNet  | src    | Resnet34  | .851  | .854     | .942  | .928   | .859    | .805      
 ----    | ss     | Resnet50  | .874  | .866     | .949  | .937   | .889    | .818      
 ----    | ms     | Resnet50  | .884  | .869     | .951  | .938   | .894    | .821      
 CPD     | src    | Resnet50  | .860  | .859     | .939  | .925   | .865    | .797      
 ----    | ss     | Resnet50  | .860  | .867     | .937  | .925   | .871    | .798      
 ----    | ms     | Resnet50  | .866  | .871     | .941  | .928   | .876    | .809      
 PoolNet | src    | Resnet50  | .871  | .863     | .944  | .931   | .880    | .808      
 ----    | ss     | Resnet50  | .865  | .867     | .939  | .931   | .877    | .794      
 ----    | ms     | Resnet50  | .869  | .862     | .943  | .928   | .877    | .806      
 EGNet   | src    | Resnet50  | .880  | .865     | .947  | .934   | .889    | .815      
 ----    | ss     | Resnet50  | .870  | .863     | .946  | .930   | .879    | .811      
 ----    | ms     | Resnet50  | .871  | .866     | .949  | .930   | .887    | .815      
 SCRN    | src    | Resnet50  | .867  | .877     | .950  | .934   | .888    | .811      
 ----    | ss     | Resnet50  | .878  | .869     | .943  | .932   | .881    | .807      
 ----    | ms     | Resnet50  | .875  | .873     | .944  | .932   | .886    | .812      
 GCPA    | src    | Resnet50  | .876  | .869     | .948  | .938   | .888    | .812      
 ----    | ss     | Resnet50  | .864  | .869     | .945  | .933   | .886    | .801      
 ----    | ms     | Resnet50  | .867  | .874     | .945  | .936   | .892    | .812      
 ITSD    | src    | Resnet50  | .876  | .872     | .946  | .935   | .885    | .821      
 ----    | ss     | Resnet50  | .874  | .870     | .943  | .932   | .878    | .807      
 ----    | ms     | Resnet50  | .873  | .875     | .946  | .933   | .887    | .816      
 MINet   | src    | Resnet50  | .879  | .867     | .947  | .935   | .884    | .810      
 ----    | ss     | Resnet50  | .867  | .873     | .940  | .929   | .881    | .802      
 ----    | ms     | Resnet50  | .871  | .874     | .945  | .935   | .890    | .819       
