# Data download

1. Download Data

* download by link
    >   download and unzip under folder named 'data'
    >
    >   [data.zip](https://drive.google.com/uc?id=1Gs9Per_unwdmlXufDxmYEgLAve0ep8Xx)

* download by shell
```shell
wget https://drive.google.com/uc?id=1Gs9Per_unwdmlXufDxmYEgLAve0ep8Xx -O data.zip
unzip -d data data.zip
```

# Conda Setting

```shell script
curl -O https://repo.anaconda.com/archive/Anaconda3-2019.10-Linux-x86_64.sh
sha256sum Anaconda3-2019.10-Linux-x86_64.sh
bash Anaconda3-2019.10-Linux-x86_64.sh

conda create -n maxwellfdfd-controlgan python=3.10
conda activate maxwellfdfd-controlgan
pip install -r requirements.txt
```

 
# Run script 

* Train
    - wgan
    ```shell script
    python wgan.py 
    ```
 
    - conditional gan (cgan)
    ```shell script
    python cgan.py
    ```
  
    - controllable gan (controlgan)
    ```shell script
    python controlgan.py
    ```


* Test
    ```shell script
    python result_test.py 
    ```
  

* Sample image to percent match, truth csv

    ```shell script
    python result_box_plot.py -fn ./logs/wgan
    ```

## To generate the data please visit the following GitHub URL : 
https://github.com/wonderit/maxwellfdfd

## Reference
https://github.com/BrainJellyPie/ControlGAN

https://github.com/igul222/improved_wgan_training

https://github.com/wonderit/maxwellfdfd-ai

https://github.com/wonderit/ControlGAN
