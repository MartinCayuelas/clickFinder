# clickFinder

## Prediction of clicks

This project is being carried out by Martin **Cayuelas**, Nicolas **Guary**, Nathan **Guillaud** and Théo **Ponthieu**.

### Get And Run the project
To run the project
```shell script
git clone https://github.com/MartinCayuelas/clickFinder.git
cd clickFinder
sbt assembly
mv ./target/scala-2.12/clickFinder-assembly-0.1.jar clickFinder.jar
```
⚠️ Do not remove or modify the model folder!
#### Predict
⚠️ If you have a `predictions` folder, please delete it before running the prediction.

To predict 
```shell script
java -jar clickFinder.jar predict nameFile
```

To predict 1000 rows
```shell script
java -jar clickFinder.jar predict1000 nameFile
```

#### Train
To train
```shell script
java -jar clickFinder.jar train nameFile
```

#### Result
The prediction results can be found in the ```data/predictions/prediction/part-00000-xxxxxx.csv``` folder