# TNT-Trajectory-Predition

An python implementation of 

[TNT: Target-driveN Trajectory Prediction](https://arxiv.org/abs/2008.08294#:~:text=TNT%20has%20three%20stages%20which,state%20sequences%20conditioned%20on%20targets.)

and

[VectorNet: Encoding HD Maps and Agent Dynamics from Vectorized Representation](https://arxiv.org/abs/2005.04259)

### Create Your Own Branch

Pull this repository to your local disk:
```
git clone https://github.com/Henry1iu/TNT-Trajectory-Predition.git
```

Create your own branch and set your up stream using push:
``` 
git checkout -b XX
``` 
(Change "XX" to any name you want.)
```
git push -u origin XX
``` 
(Change "XX" to the name you specified for your branch.)

### Update Your Code

After your finish the implementation of a module, push your local modification to your branch in the remote repository:
```
git push 
```
(Remember to add the changes and commit them to your local repository before push.)

Your can check your current local branch via ```git branch```.

**Remember** you can only push your modification to your own branch in this github repository, and don't touch the code in "main" branch.

### Get Update from Main Branch

Once the "main" branch is updated, you can get the update changes by:
```
git checkout main
git pull upstream main
git checkout XX
git merge main
```
The changes should be automaticly merged to your own branch.
If you find there is conflict, ask me for help. 

### Prerequisite
* python==3.6
* pytorch==1.4.0
* torch-geometric==1.5.0
* argoverse-api
* pandas==1.0.0

### TODO
Data-related:
- [ ] Data Loading
- [ ] Data Pre-processing

Model-related:
- [ ] Add the GNN network in the implementation
- [ ] Add the auxiliary recovery prediction in training

Training-related:
- [ ] Implement a trainer
- [ ] Implement the loss function with the auxiliary loss