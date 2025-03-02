

## Introduction

This project  is a novel approach for integrating cybersecurity vulnerability information from multiple sources.   



## Installation



### **1. Install DGL**

Due to the requirement for a higher version of DGL, the installation via `pip` or `conda` may not work. Therefore, you need to install DGL directly **from the source**. Please follow the instructions based on your operating system.

#### **For Windows Users Only: Configure Build Environment**  

Before installing DGL, **Windows users** must set up the required build environment:

   - **Visual Studio**: Download and install [Visual Studio](https://visualstudio.microsoft.com/) with the **Desktop development with C++** workload.
   - **CMake**: Download and install [CMake](https://cmake.org/download/).

Once the environment is ready, follow the **official DGL installation guide** for your operating system:  
🔗 [DGL Installation Guide](https://www.dgl.ai/dgl_docs/en/2.4.x/install/index.html)

For **Linux/macOS users**, no additional setup is required—just follow the installation guide directly.

### **2. Install Other Required Packages**

```bash
pip install fasttext-wheel

conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=12.1 -c pytorch -c nvidia

pip install -r requirements.txt
```



## Usage

To run the model, use the following command. Ｍodify the paths and parameters as needed based on  datasets and requirements.

```
python code/main.py --cp=[path to corpus] --dp=[path to dataset] --tp=[path to text embedding model] --heterogeneous=1 --mask=1
```

#### **Examples**

##### **Running the Model on default CERT-NVD Dataset**

```
python code/main.py --dp=../data/CERT-NVD/ --batch-size 128 --lr=0.02 --n-epoch 20
```

##### **Running the Model on default SF-NVD Dataset**

```
python code/main.py --dp=data/SF-NVD/ --batch-size 128 --lr=0.025 --n-epoch 30
```



## **Dataset**

It is evaluated on cybersecurity entity alignment datasets built from:

- **National Vulnerability Database (NVD)**
- **ICS-CERT Advisories**
- **SecurityFocus (SF)**

The datasets include manually annotated vulnerabilities to improve alignment accuracy.



## **Citation**

If you use it in your research, please cite the following paper:



## **Contact**

For questions or contributions, please contact: lihuawang9999@gmail.com

