# Data download

1. Download Data

* download by link
    >   download and unzip under folder named 'data'
    >
    >   [data.zip](https://drive.google.com/uc?id=14-Bl89OzRtLM1MCW2H81Xvivq8EvTrmB)

# Conda Setting

```shell script
curl -O https://repo.anaconda.com/archive/Anaconda3-2019.10-Linux-x86_64.sh
sha256sum Anaconda3-2019.10-Linux-x86_64.sh
bash Anaconda3-2019.10-Linux-x86_64.sh

conda create -n ControlGAN python=3.7
conda activate ControlGAN
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
