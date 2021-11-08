# Side face detection api
## CKM VIGIL PVT. LTD.

## 🌟**Target**

The target of this project is to detect characteristics of the side face in the image like displaying the keypoints in the image, displaying the different side face characteristics  
    
    
---
## :star2: **Started With**

We started with a [Mediapipe Facemesh](https://google.github.io/mediapipe/solutions/face_mesh.html) Model which is giving us 468 key points on the face in the image, then with the help of these obtained key points, other results can be displayed.


---
## :star2: **Overview and documentation**
#### We will find 468 key points of an image and show them on the image.


#### Various helper functions are employed to achieve the desired results. These functions are:

- `def extractCoordinates(results, landmark_number)` 

- `keypoints = image.copy()`

- `def drawArrow(image, start, end)`

- `def findAngle(start, end)`

- `def chin_ratio1(a,b,c):`

- `def lip_ratio1(a,b,c):`

- `def drawCircle(image, location, i=1):`

- `def Zangle(image, start,end,k):`
<br>



## :star: Results
  
- Sample image
  
  ![exampleimg]()
<br/>








- image showing keypoints:

  ![keypoints]()
<br/>

- image showing side face results:

 ![z angle]()

---

## :star2: **Modules and Pakages**
    
- *Modules Used*
  1. streamlit
  1. pillow
  1. deepface
  1. pandas
  1. scikit-image
  1. opencv-python
  1. mediapipe
  
 - *Pakages Used*
    1. freeglut3-dev
    1. libgtk2.0-dev      
    

---
## :star2: **Repository content**
- *streamlit_app.py* - It contain the main code of the project which is in python.

- *pakages.txt* - It contains the name of the pakages used .

- *requirements.txt* - It contains the names of the modules used to the code .

- *setup.sh* - streamlit configuration file.




---
## :star2: **Live working**
- The whole project is deployed at _Streamlit_ that turns data script to shareable web apps for python.

- [Project demo](https://share.streamlit.io/ckmvigil/faceapi_v2/main)
---

## :star2: **Contributors**
-  *Team Lead*: Vibhanshu Jain, Dhruv Makwana
-  *Members*: Amandeep Singh, Anirudh Bhasin, Abijit Singh, Damandeep Singh, Prakhar Gupta
-  *Project Lead*: Chalavadi Vishnu, Sai Chandra Teja
---
