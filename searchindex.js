Search.setIndex({envversion:46,filenames:["auto_gallery/astro/galaxy_deconvolution","auto_gallery/gallery","auto_gallery/mri/cartesian_reconstruction","auto_gallery/mri/non_cartesian_reconstruction","auto_gallery/mri/p_mri_cartesian_reconstruction","auto_gallery/structure","generated/documentation","generated/installation","generated/pysap","generated/pysap.apps","generated/pysap.base","generated/pysap.base.exceptions","generated/pysap.base.image","generated/pysap.base.io","generated/pysap.base.loaders","generated/pysap.base.loaders.fits","generated/pysap.base.loaders.loader_base","generated/pysap.base.loaders.nifti","generated/pysap.base.loaders.numpy_binary","generated/pysap.base.observable","generated/pysap.base.transform","generated/pysap.base.utils","generated/pysap.configure","generated/pysap.data","generated/pysap.extensions","generated/pysap.extensions.formating","generated/pysap.extensions.tools","generated/pysap.extensions.transform","generated/pysap.extensions.wrapper","generated/pysap.plotting","generated/pysap.plotting.image","generated/pysap.plotting.transform","generated/pysap.plotting.utils","generated/pysap.plugins","generated/pysap.plugins.astro","generated/pysap.plugins.astro.deconvolve","generated/pysap.plugins.astro.deconvolve.deconvolve","generated/pysap.plugins.astro.deconvolve.linear","generated/pysap.plugins.astro.deconvolve.wavelet_filters","generated/pysap.plugins.mri","generated/pysap.plugins.mri.parallel_mri","generated/pysap.plugins.mri.parallel_mri.extract_sensitivity_maps","generated/pysap.plugins.mri.parallel_mri.gradient","generated/pysap.plugins.mri.parallel_mri.proximity","generated/pysap.plugins.mri.parallel_mri.reconstruct","generated/pysap.plugins.mri.parallel_mri.utils","generated/pysap.plugins.mri.reconstruct","generated/pysap.plugins.mri.reconstruct.cost","generated/pysap.plugins.mri.reconstruct.fourier","generated/pysap.plugins.mri.reconstruct.gradient","generated/pysap.plugins.mri.reconstruct.linear","generated/pysap.plugins.mri.reconstruct.noise","generated/pysap.plugins.mri.reconstruct.reconstruct","generated/pysap.plugins.mri.reconstruct.reweight","generated/pysap.plugins.mri.reconstruct.utils","generated/pysap.utils","index"],objects:{"pysap.base":{exceptions:[11,7,0,"-"],image:[12,7,0,"-"],io:[13,7,0,"-"],observable:[19,7,0,"-"],transform:[20,7,0,"-"],utils:[21,7,0,"-"]},"pysap.base.exceptions":{Exception:[11,8,1,""],Sparse2dConfigurationError:[11,8,1,""],Sparse2dError:[11,8,1,""],Sparse2dRuntimeError:[11,8,1,""]},"pysap.base.image":{Image:[12,9,1,""]},"pysap.base.image.Image":{dtype:[12,10,1,""],modified:[12,11,1,""],ndim:[12,10,1,""],scroll_axis:[12,10,1,""],shape:[12,10,1,""],show:[12,11,1,""],spacing:[12,10,1,""]},"pysap.base.io":{get_loader:[13,12,1,""],get_saver:[13,12,1,""],load:[13,12,1,""],save:[13,12,1,""]},"pysap.base.loaders":{fits:[15,7,0,"-"],loader_base:[16,7,0,"-"],nifti:[17,7,0,"-"],numpy_binary:[18,7,0,"-"]},"pysap.base.loaders.fits":{FITS:[15,9,1,""]},"pysap.base.loaders.fits.FITS":{allowed_extensions:[15,10,1,""],load:[15,11,1,""],save:[15,11,1,""]},"pysap.base.loaders.loader_base":{LoaderBase:[16,9,1,""]},"pysap.base.loaders.loader_base.LoaderBase":{allowed_extensions:[16,10,1,""],can_load:[16,11,1,""],can_save:[16,11,1,""],load:[16,11,1,""],save:[16,11,1,""]},"pysap.base.loaders.nifti":{NIFTI:[17,9,1,""]},"pysap.base.loaders.nifti.NIFTI":{allowed_extensions:[17,10,1,""],load:[17,11,1,""],save:[17,11,1,""]},"pysap.base.loaders.numpy_binary":{npBinary:[18,9,1,""]},"pysap.base.loaders.numpy_binary.npBinary":{allowed_extensions:[18,10,1,""],load:[18,11,1,""],save:[18,11,1,""]},"pysap.base.observable":{Observable:[19,9,1,""],SignalObject:[19,9,1,""]},"pysap.base.observable.Observable":{add_observer:[19,11,1,""],allowed_signals:[19,10,1,""],notify_observers:[19,11,1,""],remove_observer:[19,11,1,""]},"pysap.base.transform":{MetaRegister:[20,9,1,""],WaveletTransformBase:[20,9,1,""]},"pysap.base.transform.MetaRegister":{REGISTRY:[20,10,1,""]},"pysap.base.transform.WaveletTransformBase":{analysis:[20,11,1,""],analysis_data:[20,10,1,""],analysis_header:[20,10,1,""],band_at:[20,11,1,""],bands_shapes:[20,13,1,""],data:[20,10,1,""],info:[20,10,1,""],show:[20,11,1,""],synthesis:[20,11,1,""]},"pysap.base.utils":{monkeypatch:[21,12,1,""],with_metaclass:[21,12,1,""]},"pysap.configure":{info:[22,12,1,""]},"pysap.data":{ResumeURLOpener:[23,9,1,""],copy_file:[23,12,1,""],download_file:[23,12,1,""],get_sample_data:[23,12,1,""],md5_sum_file:[23,12,1,""],progress_bar:[23,12,1,""]},"pysap.data.ResumeURLOpener":{http_error_206:[23,11,1,""]},"pysap.extensions":{formating:[25,7,0,"-"],tools:[26,7,0,"-"],transform:[27,7,0,"-"],wrapper:[28,7,0,"-"]},"pysap.extensions.formating":{flatten_decimated_1_bands:[25,12,1,""],flatten_decimated_3_bands:[25,12,1,""],flatten_decimated_feauveau:[25,12,1,""],flatten_undecimated_n_bands:[25,12,1,""],flatten_vector:[25,12,1,""],get_hb:[25,12,1,""],get_hbl:[25,12,1,""],get_hbr:[25,12,1,""],get_hl:[25,12,1,""],get_hr:[25,12,1,""],get_ht:[25,12,1,""],get_htl:[25,12,1,""],get_htr:[25,12,1,""],inflated_decimated_1_bands:[25,12,1,""],inflated_decimated_3_bands:[25,12,1,""],inflated_decimated_feauveau:[25,12,1,""],inflated_undecimated_n_bands:[25,12,1,""],inflated_vector:[25,12,1,""],set_hb:[25,12,1,""],set_hbl:[25,12,1,""],set_hbr:[25,12,1,""],set_hl:[25,12,1,""],set_hr:[25,12,1,""],set_ht:[25,12,1,""],set_htl:[25,12,1,""],set_htr:[25,12,1,""]},"pysap.extensions.tools":{mr_deconv:[26,12,1,""],mr_filter:[26,12,1,""],mr_recons:[26,12,1,""],mr_transform:[26,12,1,""]},"pysap.extensions.transform":{BsplineWaveletTransformATrousAlgorithm:[27,9,1,""],DecompositionOnScalingFunction:[27,9,1,""],FastCurveletTransform:[27,9,1,""],FeauveauWaveletTransform:[27,9,1,""],FeauveauWaveletTransformWithoutUndersampling:[27,9,1,""],HaarWaveletTransform:[27,9,1,""],HalfPyramidalTransform:[27,9,1,""],ISAPWaveletTransformBase:[27,9,1,""],IsotropicAndCompactSupportWaveletInFourierSpace:[27,9,1,""],LineColumnWaveletTransform1D1D:[27,9,1,""],LinearWaveletTransformATrousAlgorithm:[27,9,1,""],MallatWaveletTransform79Filters:[27,9,1,""],MeyerWaveletsCompactInFourierSpace:[27,9,1,""],MixedHalfPyramidalWTAndMedianMethod:[27,9,1,""],MixedWTAndPMTMethod:[27,9,1,""],MorphologicalMedianTransform:[27,9,1,""],MorphologicalMinmaxTransform:[27,9,1,""],MorphologicalPyramidalMinmaxTransform:[27,9,1,""],NonOrthogonalUndecimatedTransform:[27,9,1,""],OnLine44AndOnColumn53:[27,9,1,""],OnLine53AndOnColumn44:[27,9,1,""],PyramidalBsplineWaveletTransform:[27,9,1,""],PyramidalLaplacian:[27,9,1,""],PyramidalLinearWaveletTransform:[27,9,1,""],PyramidalMedianTransform:[27,9,1,""],PyramidalWaveletTransformInFourierSpaceAlgo1:[27,9,1,""],PyramidalWaveletTransformInFourierSpaceAlgo2:[27,9,1,""],UndecimatedBiOrthogonalTransform:[27,9,1,""],UndecimatedDiadicWaveletTransform:[27,9,1,""],UndecimatedHaarTransformATrousAlgorithm:[27,9,1,""],WaveletTransformInFourierSpace:[27,9,1,""],WaveletTransformViaLiftingScheme:[27,9,1,""]},"pysap.extensions.transform.ISAPWaveletTransformBase":{decimated:[27,13,1,""],undecimated:[27,13,1,""]},"pysap.extensions.wrapper":{Sparse2dWrapper:[28,9,1,""]},"pysap.plotting":{image:[30,7,0,"-"],transform:[31,7,0,"-"],utils:[32,7,0,"-"]},"pysap.plotting.image":{plot_data:[30,12,1,""]},"pysap.plotting.transform":{plot_transform:[31,12,1,""]},"pysap.plotting.utils":{histogram:[32,12,1,""],scaling:[32,12,1,""]},"pysap.plugins.astro.deconvolve":{deconvolve:[36,7,0,"-"],linear:[37,7,0,"-"],wavelet_filters:[38,7,0,"-"]},"pysap.plugins.astro.deconvolve.deconvolve":{get_weights:[36,12,1,""],psf_convolve:[36,12,1,""],sparse_deconv_condatvu:[36,12,1,""]},"pysap.plugins.astro.deconvolve.linear":{WaveletConvolve2:[37,9,1,""]},"pysap.plugins.astro.deconvolve.wavelet_filters":{get_cospy_filters:[38,12,1,""]},"pysap.plugins.mri.parallel_mri":{extract_sensitivity_maps:[41,7,0,"-"],gradient:[42,7,0,"-"],proximity:[43,7,0,"-"],reconstruct:[44,7,0,"-"],utils:[45,7,0,"-"]},"pysap.plugins.mri.parallel_mri.extract_sensitivity_maps":{extract_k_space_center:[41,12,1,""],get_Smaps:[41,12,1,""],gridding_2d:[41,12,1,""]},"pysap.plugins.mri.parallel_mri.gradient":{Grad2D_pMRI:[42,9,1,""],Grad2D_pMRI_analysis:[42,9,1,""],Grad2D_pMRI_synthesis:[42,9,1,""]},"pysap.plugins.mri.parallel_mri.gradient.Grad2D_pMRI":{get_cost:[42,11,1,""]},"pysap.plugins.mri.parallel_mri.proximity":{Threshold:[43,9,1,""]},"pysap.plugins.mri.parallel_mri.proximity.Threshold":{get_cost:[43,11,1,""],op:[43,11,1,""]},"pysap.plugins.mri.parallel_mri.reconstruct":{sparse_rec_condatvu:[44,12,1,""],sparse_rec_fista:[44,12,1,""]},"pysap.plugins.mri.parallel_mri.utils":{check_lipschitz_cst:[45,12,1,""],function_over_maps:[45,12,1,""],prod_over_maps:[45,12,1,""]},"pysap.plugins.mri.reconstruct":{cost:[47,7,0,"-"],fourier:[48,7,0,"-"],gradient:[49,7,0,"-"],linear:[50,7,0,"-"],noise:[51,7,0,"-"],reconstruct:[52,7,0,"-"],reweight:[53,7,0,"-"],utils:[54,7,0,"-"]},"pysap.plugins.mri.reconstruct.cost":{DualGapCost:[47,9,1,""]},"pysap.plugins.mri.reconstruct.fourier":{FFT2:[48,9,1,""],FourierBase:[48,9,1,""],NFFT2:[48,9,1,""]},"pysap.plugins.mri.reconstruct.fourier.FFT2":{adj_op:[48,11,1,""],op:[48,11,1,""]},"pysap.plugins.mri.reconstruct.fourier.FourierBase":{adj_op:[48,11,1,""],op:[48,11,1,""]},"pysap.plugins.mri.reconstruct.fourier.NFFT2":{adj_op:[48,11,1,""],op:[48,11,1,""]},"pysap.plugins.mri.reconstruct.gradient":{GradAnalysis2:[49,9,1,""],GradSynthesis2:[49,9,1,""]},"pysap.plugins.mri.reconstruct.linear":{Wavelet2:[50,9,1,""]},"pysap.plugins.mri.reconstruct.linear.Wavelet2":{adj_op:[50,11,1,""],l2norm:[50,11,1,""],op:[50,11,1,""]},"pysap.plugins.mri.reconstruct.noise":{sigma_mad_sparse:[51,12,1,""]},"pysap.plugins.mri.reconstruct.reconstruct":{sparse_rec_condatvu:[52,12,1,""],sparse_rec_fista:[52,12,1,""]},"pysap.plugins.mri.reconstruct.reweight":{mReweight:[53,9,1,""]},"pysap.plugins.mri.reconstruct.reweight.mReweight":{reweight:[53,11,1,""]},"pysap.plugins.mri.reconstruct.utils":{condatvu_logo:[54,12,1,""],convert_locations_to_mask:[54,12,1,""],convert_mask_to_locations:[54,12,1,""],fista_logo:[54,12,1,""],flatten:[54,12,1,""],unflatten:[54,12,1,""]},"pysap.utils":{TempDir:[55,9,1,""],load_image:[55,12,1,""],load_transform:[55,12,1,""],logo:[55,12,1,""],save_image:[55,12,1,""]},pysap:{configure:[22,7,0,"-"],data:[23,7,0,"-"],utils:[55,7,0,"-"]}},objnames:{"0":["np","module","Python module"],"1":["np","exception","Python exception"],"10":["py","attribute","Python attribute"],"11":["py","method","Python method"],"12":["py","function","Python function"],"13":["py","classmethod","Python class method"],"2":["np","class","Python class"],"3":["np","attribute","Python attribute"],"4":["np","method","Python method"],"5":["np","function","Python function"],"6":["np","classmethod","Python class method"],"7":["py","module","Python module"],"8":["py","exception","Python exception"],"9":["py","class","Python class"]},objtypes:{"0":"np:module","1":"np:exception","10":"py:attribute","11":"py:method","12":"py:function","13":"py:classmethod","2":"np:class","3":"np:attribute","4":"np:method","5":"np:function","6":"np:classmethod","7":"py:module","8":"py:exception","9":"py:class"},terms:{"2_2":42,"__call__":21,"__init__":21,"__version__":5,"_compat":21,"class":[6,11,12,15,16,17,18,19,20,21,23,27,28,35,37,40,41,42,43,44,46,47,48,49,50,53,55],"default":[15,20,21,23,27,30,31,32,36,37,38,43,44,50,52,53],"final":[21,41],"float":[23,32,41,42,43,44,45,50,51,52,53],"function":[2,3,4,5,6,8,9,12,19,21,23,24,25,27,29,32,42,43,44,45,46,47,55],"int":[12,20,23,27,30,31,32,36,38,44,45,48,52,54],"return":[12,13,19,20,21,22,23,25,27,32,36,38,41,42,43,44,45,48,50,51,52,53,54,55],"true":[3,4,15,18,19,23,27,31,45,47],"try":23,about:22,absolut:23,access:5,account:12,acquir:41,acquisit:[2,3,4,40,41],acquist:[2,3,4],actual:[21,25],adapt:32,add:[2,3,4,19],add_maximum_level_constraint:26,add_nois:0,add_observ:19,add_posit:[2,3,4,44,52],addit:43,adj_op:[3,48,50],adjoint:50,advantag:21,ag239446:23,algo:27,algorithm:[27,40,42,44,46,49,52],algorithm_nam:11,all:[1,5,6,8,11,16,22,23,24,25,31,33,34,35,39,40,45,46,55],allow:[5,6,8,12,19,24,56],allowed_extens:[15,16,17,18],allowed_sign:19,alpha:42,alreadi:[15,23,25],also:[2,3,4,5,21],analys:32,analysi:[5,6,8,20,42,56],analysis_data:20,analysis_head:20,ani:[23,41],anstronom:1,app:6,appli:45,applic:[5,6,8,9,56],approach:7,approxim:[27,51],arg:11,argument:45,arrai:[12,20,25,27,30,36,37,38,42,43,48,50,53,54],arri:20,art:55,ascii:[54,55],associ:[3,15,16,17,18,20,30],astropi:37,atol:[2,3,4,44,52],attribut:48,auto_gallery_jupyt:1,auto_gallery_python:1,avail:[2,3,4,20,24,27,55],available_transform:[5,55],axi:[12,30],back:21,background_model_imag:26,band:[20,25,27,31],band_at:20,band_data:20,bands_length:[20,27],bands_nam:27,bands_shap:[20,27],bar:23,bar_length:23,baseform:21,basic:21,becaus:21,bee:23,been:[5,41],best:7,between:[20,27,41,54],bin:32,binari:18,bind:[5,20,56],bit:21,bool:[15,19,23,27,31,32,36,38,44,45,52],bottom:25,bound:45,brain:[2,3,4],bsd:21,bspline:27,bsplinewavelettransformatrousalgorithm:[2,3,20,24,27],calcul:[23,42,43,48],call:[11,19,38],callabl:[21,27,45],can:[5,25],can_load:16,can_sav:16,carri:[2,3],cartesian:1,cartesian_reconstruct:2,cast:13,catesian:48,center:[40,41],chang:32,channel:41,check_lip:42,check_lipschitz_cst:45,classic:25,classmethod:[20,27],classs:[40,42,46,49],clean:0,clobber:[15,18],clone:7,closer:21,coars:38,code:[0,1,2,3,4,5,43],coef_detection_method:26,coeff:50,coeffici:[20,25,27,37,44,48,50,52],coil:41,column:27,com:[5,7],come:21,command:28,command_nam:11,common:1,compact:27,complex:[20,48],comput:[27,31,32,42,44,45,50],concern:25,conda:[35,36],condat:[2,3,4,36,40,44,46,52,54],condatvu_logo:54,configur:[5,8,22],consid:32,consider_missing_data:26,consist:[44,52],constant:45,constraint:45,contain:[6,12,20,24,25,32,35,37,38,40,41,42,45,46,49,50,52],context:5,contribut:7,control:23,convent:27,converg:[44,52],convert:54,convert_locations_to_mask:54,convert_mask_to_loc:[2,3,4,54],convolut:[36,37],convolv:[0,36,50],copi:23,copy_fil:23,correspond:25,cosmostat:5,cospi:38,cost:[2,3,4,42,43,44,46,47],cost_interv:47,creat:[7,23,55],credit:[0,2,3,4,5],cube:[24,25],cubic:41,cumul:32,curvelet:[25,27],custom:21,dans:54,data_dir:23,data_shap:38,data_typ:[5,12],datadir:23,datafidel:42,dataset:[2,3,4,5,6,8,10,14,23,32,52,54],dataset_nam:23,deal:[6,14],decim:[25,27],declar:[5,6,13,33],decomposit:[5,6,8,20,24,25,27,31,44,52,56],decompositiononscalingfunct:[20,24,27],decompsit:5,deconv_data:0,deconvolv:[0,6],decor:21,def:21,defin:[6,10,12,14,15,17,18,19,20,21,23,27,29,33,34,35,37,39,40,42,43,46,47,49,50],delet:23,denois:[5,6,8,24,56],densiti:[40,41],depend:[5,8,22],deriv:[31,42,44,52],describ:53,desir:[31,54,55],destin:13,detail:[6,27,41],detect_only_positive_structur:26,develop:[5,7],deviat:[44,52],diadic:27,dict:[19,20],dictionari:[12,23],dictionarybas:43,diff:27,differ:[20,27,31,46,47],differenti:[42,44],differentiablepart:42,dim:48,dimens:[12,25],direct:45,directori:[7,23],discret:48,dispali:22,displai:[12,20,30,31],domain:[41,44,48,52],dowload:23,downgrad:21,download:[0,1,2,3,4,5,8,23],download_fil:23,download_fnam:23,dtype:[5,12,13,50],dual:[2,3,4,44,47,52],dualgapcost:47,dummi:[19,21],dure:[5,44,52],dynam:32,each:[20,25,27,36,51,53],ecept:19,eeach:20,effect:0,element:45,elgueddari:[2,3,4],emb:5,emit:19,encod:25,enough:6,entre:45,env:28,epsilon:26,equal:[32,45],equat:42,errcod:23,errmsg:23,error:[11,23],estim:[41,44,46,51,52,53],etc:21,event:[12,19],everi:45,everybodi:5,exampl:0,except:[10,11,13,19],execut:[6,9],exist:[15,23],expect:[44,52],explan:21,express:[44,52],extend:[5,21],extens:[6,16,20],extra_factor:43,extract:[40,41],extract_k_space_cent:41,extract_sensitivity_map:[4,40,41],factor:[27,36,43,53],fail:[11,23],fals:[2,3,4,23,26,28,31,32,36,38,42,44,52,55],farren:0,fasl:19,fastcurvelettransform:[5,20,24,27],feauveau:[25,27],feauveauwavelettransform:[20,24,27],feauveauwavelettransformwithoutundersampl:[20,24,27],fender:55,fft2:[2,4,48],fft:4,fftpack:[2,3,4],fftw:52,figur:4,file:[13,15,16,23],filenam:15,fill:[2,3,4,20],filter:[27,35,36,37,38,50],fine:23,first_detection_scal:26,first_guess_file_nam:26,fista_logo:54,fit:[5,14,15,24,25],flat_imag:26,flatten:[24,25,54],flatten_decimated_1_band:25,flatten_decimated_3_band:25,flatten_decimated_feauveau:25,flatten_fct:27,flatten_undecimated_n_band:25,flatten_vector:25,float32:13,fname:23,follow:[12,42],font:[54,55],foo:21,form:21,format:[24,25,27],formtyp:21,found:13,fourier:[3,27,42,44,46,48,52],fourier_op:[3,4,42,49],fourierbas:48,fourrier:48,free:[2,3,4],from:[0,2,3,4,5,19,20,21,24,25,27,40,41,43,51],full:6,func:19,function_over_map:[4,45],further:6,galaxy_deconvolut:0,galleri:[0,1,2,3,4,5,6],gap:47,gaussian:[2,3,4],gener:[0,1],get:[2,3,4,12,20,23,35,36,38],get_cospy_filt:38,get_cost:[4,42,43,44],get_hb:25,get_hbl:25,get_hbr:25,get_hl:25,get_hr:25,get_ht:25,get_htl:25,get_htr:25,get_load:13,get_sample_data:[0,2,3,4,5,23],get_sav:13,get_smap:[4,41],get_weight:36,gettng:42,git:7,github:[5,7],give:[5,6],given:[19,20,25,41,45],grad2d_pmri:[4,42],grad2d_pmri_analysi:42,grad2d_pmri_synthesi:[4,42],grad:42,grad_op:51,gradanalysis2:49,gradbas:44,gradient:[4,40,42,44,46,49,51],gradient_op:[4,44],gradient_op_cd:4,gradsynthesis2:49,graphic:[6,9],grid:41,griddata:41,gridding_2d:41,grigi:[2,3,5],guidelin:6,haar:27,haarwavelettransform:[20,24,27],had:41,half:[25,27],halfpyramidaltransform:[20,24,27],hard:43,have:[5,12,25],header:[20,23],heavili:41,here:[2,3,4],high:[40,41],hist_im:32,histogram:32,hold:[20,27],home:[4,23],host:5,how:1,hpmt:27,http:[5,7],http_error_206:23,hyperparamet:[44,52],icf_file_nam:26,icf_fwhm:26,idea:21,ident:[44,52],ifft2:2,ifftshift:2,iform:20,illustr:5,image_nam:4,image_ob:0,image_path:55,image_rec0:[2,3],image_rec:[0,2,3,4],image_shap:42,img:48,img_shap:[41,54],implement:53,in_imag:26,in_mr_fil:26,in_psf:26,includ:27,index:20,inflat:25,inflated_decimated_1_band:25,inflated_decimated_3_band:25,inflated_decimated_feauveau:25,inflated_undecimated_n_band:25,inflated_vector:25,info:[5,20,22],inform:[20,22,41,54],initi:[44,52],initial_cost:47,input:[2,3,4,20,23,32,36,42,43,45,48,50,52,54],instal:5,instanc:[13,20,31,42,44,51,52],instanti:21,intal:7,intens:32,interfac:[5,19],intern:21,interpol:41,introduc:21,introductori:1,invers:48,invl:42,ipynb:[0,2,3,4,5],is_decim:27,isapwavelettransformbas:27,iso:41,iso_shap:27,isotrop:27,isotropicandcompactsupportwaveletinfourierspac:[20,24,27],iter:[2,3,4,36,44,52],itself:21,jinja2:21,jupyt:[0,1,2,3,4,5],justifi:23,k_space:41,k_space_ref:4,keep:38,klass:21,kspace_data:[2,4],kspace_loc:[2,3,4],kspace_mask:2,kspace_ob:3,kwarg:[11,12,19,20,27],l2norm:50,lab:5,lambda_init:[2,3,4,44,52],laplacian:27,last:45,latest:7,layer:25,lead:25,left:25,length:[20,27,45],level:[21,23,44,52],librari:[5,6,8,24,56],licens:21,lift:27,like:[21,25],linalg:4,line:27,linear:[4,27,35,37,41,42,44,46,50,51,53],linear_op:[4,42,44,47,49,51,53],linearbas:44,linearwavelettransformatrousalgorithm:[20,24,27],linecolumnwavelettransform1d1d:[20,24,27],lipschitz:45,lipschitz_cst:45,lipschitzien:45,lischitz:45,list:[8,20,27,31,45,51,54,55],load:[0,2,3,4,8,13,15,16,17,18,23,55],load_imag:55,load_transform:[5,55],loader:[6,13],loader_bas:[14,16],loaderbas:16,loadmat:4,local:23,locat:[2,3,4,41,54],logo:[54,55],loubnaelgueddari:4,lower_cut:32,machin:5,mad:[44,51,52],mai:6,make:21,mallat:27,mallatwavelettransform79filt:[20,24,27],mandatori:23,mani:5,map:[40,41,42,45],mask:[2,3,4,44,48,52,54],mask_file_nam:26,mat:4,math:0,matplotlib:4,matrix:[12,42,48,52,54],max:[20,27],max_it:[2,3,4],max_nb_of_it:[2,3,4,44,45,52],maximum:[2,3,4,36,44,52],maxsiz:23,md5:23,md5_sum_fil:23,mean:23,meas_mid41_csgre_ref_os1_fid14687:4,measur:[2,3,4],mediac:[5,56],median:27,messag:11,meta:21,metaclass:[20,21],metadata:[2,3,4,5,12,15,16,17,18],metaregist:20,meth:21,method:[15,16,17,18,21,27,32,35,37,38,41,42,43,45,48,50],methodnam:21,meyer:27,meyerwaveletscompactinfourierspac:[20,24,27],ming2017:53,ming:53,minmax:27,minut:[0,2,3,4,5],mix:27,mixedhalfpyramidalwtandmedianmethod:[20,24,27],mixedwtandpmtmethod:[20,24,27],mode:[4,41],modifi:12,modopt:[0,40,43],modul:[6,8,9,10,14,20,21,22,23,24,25,27,29,32,33,34,35,37,38,39,40,41,42,45,46,49,50,55],moment:[44,52],monkeypatch:21,more:[2,3,4,27,41],morpholog:27,morphologicalmediantransform:[20,24,27],morphologicalminmaxtransform:[20,24,27],morphologicalpyramidalminmaxtransform:[20,24,27],mr_deconv:26,mr_filter:26,mr_recon:26,mr_tansform:20,mr_transform:26,mreweight:53,mro:21,multipl:[25,43],multiview:31,must:[23,45],n_iter:36,n_reweight:36,n_scale:38,n_x:41,n_y:41,name:[21,23,27,38,44,52,55],nb_band:27,nb_band_per_scal:[20,25,27],nb_of_reweight:[2,3,4,44,52],nb_scale:[2,3,4,5,20,25,27,44,50,52],nbin:32,nbiter:26,ndarrai:[12,13,20,25,27,36,37,38,41,42,43,44,45,48,50,52,53,54],ndim:12,nearest:41,necessarli:[48,54],need:[21,23,40,45],neuroimag:1,neurospin:[5,7],nfft2:[3,48],nfftw:52,nice:54,nifti:[2,3,5,14,17],nii:17,niter_sigma_clip:26,no_auto_shift_max_psf:26,nois:[0,2,3,4,44,46,51,52],noisi:0,non:1,non_cartesian:[3,52],non_cartesian_reconstruct:3,none:[2,3,4,11,12,13,20,21,23,26,28,31,41,42,44,47,52],nonorthogonalundecimatedtransform:[20,24,27],norm:[4,50],normal:32,normalize_imag:32,note:23,notebook:[0,1,2,3,4,5],noth:23,notif:19,notifi:19,notify_observ:19,now:[2,3,4,5],npbinari:18,npy:18,nsigma:26,number:[2,3,4,27,32,36,38,41,44,45,52],number_of_iter:26,number_of_scal:26,number_of_undecimated_scal:26,numer:53,numpi:[0,2,3,4,12,13,18],numpy_binari:[14,18],oberv:19,object:[19,21,42,44],obs_data:0,observ:[0,2,3,4,10,12,19,44,52],obtain:38,offici:7,onli:[20,44,52],online44andoncolumn53:[20,24,27],online53andoncolumn44:[20,24,27],oper:[35,37,40,42,43,44,46,48,49,50,51,52,53],optimis:[46,53],optimz:5,option:[15,20,23,30,36,37,38,41,43,44,52],order:[2,3,4,5,23],orthogon:27,otherwis:[19,50],out:[19,45],out_imag:26,out_mr_fil:26,outpath:[15,16,17,18],output:[16,41,54],over:[21,45],overid:23,overload:[40,43],overview:1,overwrit:[15,23],p_mri:[40,45],p_mri_cartesian_reconstruct:4,packag:[2,3,4,5,6],page:56,parallel:[41,42],parallel_mri:[4,6],paramet:[2,3,4,11,13,15,16,17,18,19,20,21,23,25,27,30,31,32,36,37,38,41,42,43,44,45,48,50,51,52,53,54,55],parent:28,part:[42,43,44],parti:[2,3,4],partial:23,pass:[13,20,21],path:[13,15,16,17,18,23,55],pattern:20,per:[20,25,27],pfft:[2,3,4],plai:[2,3,4],pleas:6,plot:[4,6],plot_data:30,plot_output:47,plot_transform:31,plottint:[29,32],plt:4,plugin:[0,2,3,4,6],pmt:27,point:41,posit:[44,52],pprint:5,predefin:37,previou:7,primal:[44,52,53],print:5,privid:[8,23],privileg:7,procedur:23,process:7,prod_over_map:[4,45],product:45,progress:[19,23],progress_bar:23,project:[5,7],properli:[5,8,22,24,25],provid:[5,6,9,41],proxim:[40,43,44,52],psf:[0,36],psf_convolv:36,psf_rot:36,pyplot:4,pyramid:27,pyramidalbsplinewavelettransform:[20,24,27],pyramidallaplacian:[20,24,27],pyramidallinearwavelettransform:[20,24,27],pyramidalmediantransform:[20,24,27],pyramidalwavelettransforminfourierspacealgo1:[20,24,27],pyramidalwavelettransforminfourierspacealgo2:[20,24,27],pysap:0,python:[0,1,2,3,4,5,6],rais:[13,19],randn:[2,3],random:[0,2,3,45],rather:52,ratio:[20,23],raw:[6,24,25],reach:[2,3,4],real:[20,45,48],reason:23,rec_imag:5,reconsruct:50,reconstruct:1,ref:4,refer:6,refin:[2,3,4],registeri:5,registri:20,regul_param:26,regular:[44,52],relat:[5,6,8,34,39,56],relax:[44,52],relaxation_factor:[2,3,4,44,52],releas:7,remov:[19,23],remove_observ:19,reorgan:27,replac:21,repres:20,request:[20,23,31],requir:[21,52],residu:[0,44,52],residual_file_nam:26,resolut:[27,41],respect:42,respectivelli:12,result:[27,42,45,48],resum:23,resumeurlopen:23,reusabl:5,reweight:[36,44,46,52,53],right:25,rms_map_file_nam:26,root:7,rotat:36,routin:[44,51,52],run:[0,2,3,4,5],same:[25,48],sampl:[2,3,4,23,41,44,48,52,54],sample_date_fil:23,samples_loc:[41,54],sapec:41,satisfi:45,save:[13,15,16,17,18,55],save_imag:55,saver:13,scalar:12,scale:[5,20,25,27,31,32,36,38,44,51,52,53],scale_shift:27,scheme:[2,3,4,27,41,53],scipi:[2,3,4,37,41],script:[0,2,3,4,5],scroll:[12,30],scroll_axi:[5,12,30],search:13,second:[0,2,3,4,5],see:[5,27,41],seek:44,seem:45,select:31,self:21,send:12,sensit:[40,41,42,45],sent:[19,23],sequenc:48,set:[2,3,4,5,25,32,44,52],set_hb:25,set_hbl:25,set_hbr:25,set_hl:25,set_hr:25,set_ht:25,set_htl:25,set_htr:25,setup:7,shape:[2,3,4,5,12,20,27,38,41,45,48,50,52,54],share:23,show:[0,2,3,4,5,12,20],sigma:[0,2,3,4,26,44,51,52,53],sigma_est:53,sigma_mad_spars:51,signal:[0,12,19,20],signalobject:19,simpl:[19,20],simul:0,six:21,size:[12,23,41,42,45],size_block:26,slice:[2,3,4,5],slider:31,smap:[4,41],snr:0,snr_file_nam:26,soft:43,softwar:[5,11],solut:[2,3,4,41,44,52,53],some:[2,3,4,22],someth:21,sourc:[0,1,2,3,4,5,11,12,13,15,16,17,18,19,20,21,22,23,25,26,27,28,30,31,32,36,37,38,41,42,43,44,45,47,48,49,50,51,52,53,54,55],space:[2,3,4,5,12,27,40,41,44,46,48,52],spars:[2,4,5,6,8,24,36,43,44,52,56],sparse2d:[5,6,8,11,24,26,28,56],sparse2dconfigurationerror:11,sparse2derror:11,sparse2druntimeerror:11,sparse2dwrapp:28,sparse_deconv_condatvu:[0,36],sparse_rec_condatvu:[2,3,4,44,52],sparse_rec_fista:[2,3,4,44,52],sparsiti:[5,6,8,36,44,56],specif:[2,3,4,6,20,31],sphinx:[0,1,2,3,4,5],split:[44,52],squar:[27,48,54],squarr:41,standard:[44,48,52],start:[2,3,4],statisfi:45,statu:23,std:[44,51,52],std_est:[2,3,4,44,52],std_est_method:[2,3,4,44,52],std_thr:[2,3,4,44,52],step:[7,44,52],store:[12,27,55],str:[13,15,16,17,18,19,21,22,23,27,32,37,38,43,44,50,52,54,55],strategi:[5,46,51,53],stretch:32,sub:23,subsampl:[2,4],sudo:7,suitabl:13,sum:[23,41,42],support:[27,30,44,52],support_file_nam:26,suppress_isolated_pixel:26,suppress_last_scal:26,suppress_positive_constraint:26,sure:21,synthax:55,synthesi:[5,20,42],tau:[2,3,4,44,52],teach:1,teh:41,tempdir:55,term:42,test:5,test_rang:47,than:[25,45,52],thi:[0,2,3,4,5,6,7,8,10,12,14,19,20,21,22,23,24,25,29,32,33,34,35,37,38,39,40,41,42,43,44,45,46,48,49,50,52,53,56],third:[2,3,4,45],thr:41,three:27,thresh_factor:53,thresh_typ:43,threshold:[32,36,41,43,44,52,53],through:7,thrown:11,time:[0,2,3,4,5,45],titl:23,toi:[2,3,4,5,8,23],toler:[44,47,52],tool:[6,10,21,24,26,29,40,41,45,46,54],top:25,total:[0,2,3,4,5],tranform:44,transform:[2,3,4],transform_klass:5,transform_nam:38,transfrom:55,treshold:[44,52],trf:25,trou:27,tupl:[38,41,45,48,54],tutori:[0,2,3,4,5,7],two:[20,27,45],type:[11,12,13,21,43],type_of_deconvolut:26,type_of_filt:26,type_of_lifting_transform:26,type_of_multiresolution_transform:26,type_of_nois:26,type_of_non_orthog_filt:26,undecim:[25,27],undecimatedbiorthogonaltransform:[4,20,24,27],undecimateddiadicwavelettransform:[20,24,27],undecimatedhaartransformatrousalgorithm:[20,24,27],under:32,undersampl:[27,40,41],unflatten:54,unflatten_fct:27,uniform:52,uniform_data_shap:[3,52],updat:53,uplet:[20,27,50,52,54],upper:45,url:23,use_l2_norm:26,use_second_generation_filt:26,useful:[10,21,22,29,32],user:7,util:[2,3,4,8,10,21,23,29,32,40,45,46,54,55],valid:19,valu:[41,44,52],variabl:41,varianc:53,variou:5,vector:[12,20,25,27],verbos:[2,3,4,5,20,23,26,27,28,44,47,50,52],version:5,via:[27,37],visual:5,vizualis:[6,29],walk:7,want:[2,3,4,7,23],wave_thresh_factor:36,wavelet2:[4,50],wavelet:[10,20,24,25,27,31,35,36,37,38,44,50,52,53],wavelet_filt:[35,38],wavelet_nam:[2,3,4,44,50,52],waveletconvolve2:37,wavelettransformbas:[20,31,44,52,55],wavelettransforminfourierspac:[20,24,27],wavelettransformvialiftingschem:[20,24,27],weight:[36,43,53],when:[11,19,20],where:[15,16,17,18,25,41,45],whether:12,which:[13,23,25],who:7,why:21,wich:44,wise:45,with_metaclass:21,without:[7,27,44,52],wrap:[6,24,26,28],wrapper:[24,28],write_all_band:26,write_all_bands_with_block_interp:26,write_info_probability_map:26,x_final:[2,3,4,44,52],x_new:53,x_shape:45,you:[5,7,23],your:5,zero:[2,3,4],zip:1},titles:["Galaxy Image Deconvolution","pySAP usage examples","Neuroimaging cartesian reconstruction","Neuroimaging non-cartesian reconstruction","Neuroimaging cartesian reconstruction","Anstronomic/Neuroimaging common structure overview","API documentation of PYSAP","Installing <cite>pysap</cite>","API documentation of <em>pysap</em>","API documentation of <em>pysap.apps</em>","API documentation of <em>pysap.base</em>","&lt;no title&gt;","&lt;no title&gt;","&lt;no title&gt;","API documentation of <em>pysap.base.loaders</em>","&lt;no title&gt;","&lt;no title&gt;","&lt;no title&gt;","&lt;no title&gt;","&lt;no title&gt;","&lt;no title&gt;","&lt;no title&gt;","&lt;no title&gt;","&lt;no title&gt;","API documentation of <em>pysap.extensions</em>","&lt;no title&gt;","&lt;no title&gt;","&lt;no title&gt;","&lt;no title&gt;","API documentation of <em>pysap.plotting</em>","&lt;no title&gt;","&lt;no title&gt;","&lt;no title&gt;","API documentation of <em>pysap.plugins</em>","API documentation of <em>pysap.plugins.astro</em>","API documentation of <em>pysap.plugins.astro.deconvolve</em>","&lt;no title&gt;","&lt;no title&gt;","&lt;no title&gt;","API documentation of <em>pysap.plugins.mri</em>","API documentation of <em>pysap.plugins.mri.parallel_mri</em>","&lt;no title&gt;","&lt;no title&gt;","&lt;no title&gt;","&lt;no title&gt;","&lt;no title&gt;","API documentation of <em>pysap.plugins.mri.reconstruct</em>","&lt;no title&gt;","&lt;no title&gt;","&lt;no title&gt;","&lt;no title&gt;","&lt;no title&gt;","&lt;no title&gt;","&lt;no title&gt;","&lt;no title&gt;","&lt;no title&gt;","pySAP"],titleterms:{"case":1,"import":[0,2,3,4,5],anstronom:5,api:[6,8,9,10,14,24,29,33,34,35,39,40,46],app:9,astro:[0,1,34,35],astronom:5,base:[5,10,14],cartesian:[2,3,4],check:5,common:5,condata:[2,3,4],content:1,current:7,data:[0,2,3,4,5],decompos:5,deconvolut:0,deconvolv:35,document:[6,8,9,10,14,24,29,33,34,35,39,40,46],exampl:1,extens:24,fast:5,first:5,fista:[2,3,4],galaxi:0,gener:[2,3,4],imag:[0,5],instal:7,isap:5,kspace:[2,3,4],loader:14,mri:[1,39,40,46],neuroimag:[2,3,4,5],non:3,optim:[2,3,4],overview:5,packag:7,parallel_mri:40,pip:7,plot:29,plugin:[33,34,35,39,40,46],pysap:[1,6,7,8,9,10,14,24,29,33,34,35,39,40,46,56],python:7,recompos:5,reconstruct:[2,3,4,46],search:56,stabl:7,structur:[1,5],transform:5,usag:1,version:7}})