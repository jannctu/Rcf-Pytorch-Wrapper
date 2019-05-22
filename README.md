# Rich Convolutional Network Pytorch Wrapper

This project is a pytorch wrapper for original RCF models. 



the following are the side outputs and the prediction example
![prediction example](https://raw.githubusercontent.com/jannctu/Rcf-Pytorch-Wrapper/master/img/bsds/235098.jpg)

<img src="https://github.com/jannctu/Rcf-Pytorch-Wrapper/blob/master/img/bsds/input/235098.jpg" alt="Input"
	title="Input" width="438" height="275" />
<img src="https://github.com/jannctu/Rcf-Pytorch-Wrapper/blob/master/img/bsds/235098.png" alt="Prediction"
	title="Prediction" width="438" height="275" />

# Converted Models
I download the original model from <a href="https://github.com/yun-liu/rcf">yun-liu</a> github and use this <a href="https://github.com/vadimkantorov/caffemodel2pytorch">Project</a> to convert the original models to the HDF5 File. <br/>
<br/> Here are the models that already converted into HDF5 file : 
<ul>
<li><a href="https://drive.google.com/open?id=1G3RvmFPNswrExbfBS7X9Q2NiZX8c8N7x">BSDS Converted Model</a> </li>
<li><a href="https://drive.google.com/open?id=1gxqxOyyJInQ1wPGH1TkHLh8dhx8MnEiX">NYUD Color Converted Model</a> </li> 
<li><a href="https://drive.google.com/open?id=1-jmK7MaPJ2GVS10ESFImEj4uftskkhwn">NYUD HHA Converted Model</a> </li>
</ul>

 # Evaluation 
 The accuracy is still lower than the original paper. I guess this is because the different bilinear interpolation and crop implementation between Caffe and this project. Sorry for inconvenience. <br/>
 I only compare the original models not the multi-scale versions.
 <table>
  <tr>
    <th>Dataset</th>
    <th>Ori-ODS</th> 
    <th>Our-ODS</th>
  </tr>
  <tr>
    <td>BSDS500</td>
    <td>0.806</td> 
    <td>0.789</td>
  </tr>
  <tr>
    <td>NYUD-Color</td>
    <td>0.743</td> 
    <td>0.708</td>
  </tr>
  <tr>
    <td>NYUD-HHA</td>
    <td>0.703</td> 
    <td>0.669</td>
  </tr>
  
</table>

# Related Projects
<ul>
 <li>HED (Official) : <a href="https://github.com/s9xie/hed">https://github.com/s9xie/hed</a></li>
 <li>HED (Pytorch) : <a href="https://github.com/xwjabc/hed">https://github.com/xwjabc/hed</a></li>
 <li>RCF (Official) : <a href="https://github.com/yun-liu/rcf">https://github.com/yun-liu/rcf</a></li>
 <li>RCF (Pytorch) : <a href="https://github.com/meteorshowers/RCF-pytorch">https://github.com/meteorshowers/RCF-pytorch</a></li>
</ul>

# Citation 
    @inproceedings{liu2017richer,
      title={Richer Convolutional Features for Edge Detection},
      author={Liu, Yun and Cheng, Ming-Ming and Hu, Xiaowei and Wang, Kai and Bai, Xiang},
      booktitle={IEEE conference on computer vision and pattern recognition (CVPR)},
      pages={3000--3009},
      year={2017}
    }
    
    @article{liu2019richer,
      title={Richer Convolutional Features for Edge Detection},
      author={Yun Liu and Ming-Ming Cheng and Xiaowei Hu and Jia-Wang Bian and Le Zhang and Xiang Bai and Jinhui Tang},
      journal={IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI)},
      year={2019},
      publisher={IEEE}
    }
    
