# FaceID

Face verification(face to passport) web-service.

## About

This web-service provides face verification solution. It requires face image and cropped passport image. You can read more about FaceID at LifeTech Wiki. 

## Install and run


```bash
$ git clone ...
$ cd ...
$ docker build -t faceid .
$ docker run -p 0.0.0.0:8000:8000/tcp faceid
```

* **P.S.** You can use any port you want. Just change it at app.py