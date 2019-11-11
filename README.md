# clickFinder

## Prediction of clicks

This project is being carried out by Martin **Cayuelas**, Nicolas **Guary**, Nathan **Guillaud** and Théo **Ponthieu**.

### Get And Run the project
To run the project
```shell script
git clone https://github.com/MartinCayuelas/clickFinder.git
cd clickFinder
sbt assembly
```
⚠️ Do not remove or modify the `models` folder!
#### Predict

To predict 
```shell script
java -jar clickFinder.jar predict <nameFile> <modelName (default = "randomForestModel")>
```

To predict 1000 rows
```shell script
java -jar clickFinder.jar predict1000 <nameFile>
```

#### Train
To train
```shell script
java -jar clickFinder.jar train <nameFile> <modelName>
```

#### Result
The prediction results can be found in the ```output/<fileNameGivenWithoutExtension>/part-00000-xxxxxx.csv``` folder