# TFG
En aquest repositori esta emmagatzemat tot el contingut, en l'àmbit de codi, del Treball de Fi de Grau: Descripció del contingut d’una imatge amb processament de llenguatge natural.  https://github.com/Auramch/TFG

## Execució del codi training
En la present versió del codi, per l'execució d'aquest amb paràmetres, cal modificar de forma manual el valor d'aquests. Més concretament, aquest són els de les línies 85 a la 92, ambdues incloses.
```
    batch = 512 #mida del batch
    paciencia = 3 #iteracions de paciencia
    left_epochs = 30 #número d'epoch
    learning_rate =  0.005 #valor de learning rate
    dim1 = 512 #dimensió d'entrada de la capa hidden del model fusió
    dim2 = 512 #dimensió de sortida de la capa hidden del model fusió
    project_name = "model_{}batch_{}learning_rate_{}dim1_{}dim2.pth".format(batch,learning_rate, dim1, dim2) #model a emmagatzemar     
```

En la funció main línia 293. S'ha de cridar a la funció training()

## Execució del codi testing
Per l'execució d'aquest amb paràmetres, cal modificar de forma manual el valor d'aquests. Més concretament, aquest són els de les línies 223 a la 227, ambdues incloses.
```
    batch = 128 #mida del batch
    dim1 = 1024 #dimensió d'entrada de la capa hidden del model fusió
    dim2 = 1024 #dimensió de sortida de la capa hidden del model fusió
    learning_rate = 0.005 #valor de learning rate
    model = "model_128batch_0.005learning_rate_1024dim1_1024dim2.pth" #model a carregar
 ```
 En la funció main línia 293. S'ha de cridar a la funció testing()
