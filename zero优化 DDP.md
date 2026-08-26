##### zero0优化 DDP

```
                           [D1,D2...D_half,Dhalf+1...Dn]
 									|          |
                                    |          |
                                    |          |
                                  batch1     batch2
                                    /           \
                   gpu0（完整model的layer-i层） gpu1（完整model的layer-i层）
                                    \            /
                                  forward     forward
                                  	/            \
                                 backward        backward
                                 	|              |
                               g11 g12 g13 g14   g21 g22 g23 g24
                                 	|              |
                                 	   计算全局梯度
                                    g1=(g11+g21)/2
                                    g2=(g12+g22)/2
                                    g3=(g13+g23)/2
                                    g4=(g14+g24)/2
                                    	  |
                                    reduce + scatter + 通信
                                          |
                                        zero0
                                          |          
                                  每个gpu都得到全局梯度
                                  		  |
                                  每个gpu的adam使用全局梯度更新W1 W2 W3 W4
                                  每个gpu的adam的模型参数m v都是完整的
                                       w1=g1+m1/v1*w1
                                       w2=g2+m2/v2*w2
                                       
                                       w3=g3+m3/v3*w3
                                       w4=g4+m4/v4*w4
                                          |
                                    每个gpu得到更新后的完整模型
                                    	  |
                                下一次forward更新w1 w2 w3 w4
                           
```

##### zero1优化 adam中的m v是分开的

```
gpu0中的adam只有m1 m2 v1 v2 只负责w1 w2的更新 只需要g1 g2

gpu1中的adam只有m3 m4 v3 v4 只负责w3 w4的更新 只需要g3 g4

通过reduce+scatter+通信 每个gpu也会得到完整的模型

这样每个gpu存储的adam参数减少了，显存也减少了占用
                           
                           [D1,D2...D_half,Dhalf+1...Dn]
 									|          |
                                    |          |
                                    |          |
                                  batch1     batch2
                                    /           \
                   gpu0（完整model的layer-i层 ） gpu1（完整model的layer-i层）
                                    \            /
                                  forward     forward
                                  	/            \
                                 backward        backward
                                 	|              |
                               g11 g12 g13 g14   g21 g22 g23 g24
                                 	|              |
                                 	   计算全局梯度
                                    g1=(g11+g21)/2
                                    g2=(g12+g22)/2
                                    g3=(g13+g23)/2
                                    g4=(g14+g24)/2
                                    	  |
                                    reduce + scatter + 通信
                                          |
                                        zero1
                                        /  \      
                                      g1 g2 g3 g4 每一个光谱都得到全部梯度
                                        |    |
                                  	  gpu0	 gpu1  
                                  	   /       \
                                  	                                          		                                  w1=g1+m1/v1*w1   w3=g3+m3/v3*w3                                                                                          
                               w2=g2+m2/v2*w2   w4=g4+m4/v4*w4
                                  		  
                                  	\               /
                                      w1 w2 w3 w4
                                       /       \
									 w3 w4	   w1 w2			
                                     gpu0       gpu1
                                     	   |
                                     
                                    每个gpu得到更新后的完整模型
                                    	  |
                                下一次forward更新w1 w2 w3 w4
                           
```

##### zero2 梯度分片 在zero1中每一个GPU只需要对应的梯度

```
gpu0中的adam只有m1 m2 v1 v2 只负责w1 w2的更新 只需要g1 g2

gpu1中的adam只有m3 m4 v3 v4 只负责w3 w4的更新 只需要g3 g4

通过reduce+scatter+通信 每个gpu也会得到完整的模型

这样每个gpu存储的adam参数减少了，显存也减少了占用
                           
                           [D1,D2...D_half,Dhalf+1...Dn]
 									|          |
                                    |          |
                                    |          |
                                  batch1     batch2
                                    /           \
                      gpu0（完整model layer-i层） gpu1（完整model layer-i层）
                                    \            /
                                  forward     forward
                                  	/            \
                                 backward        backward
                                 	|              |
                               g11 g12 g13 g14   g21 g22 g23 g24
                                 	|              |
                                 	   计算全局梯度
                                    g1=(g11+g21)/2
                                    g2=(g12+g22)/2
                                    g3=(g13+g23)/2
                                    g4=(g14+g24)/2
                                    	  |
                                    reduce + scatter + 通信
                                          |
                                        zero1
                                        /  \      
                                     g4 g3 g2 g1 梯度分片
                                       |    |
                                  	  gpu0	 gpu1  
                                  	   /       \
                                  	                                          		                                  w1=g1+m1/v1*w1   w3=g3+m3/v3*w3                                                                                          
                               w2=g2+m2/v2*w2   w4=g4+m4/v4*w4
                                  		  
                                  	\               /
                                      w1 w2 w3 w4
                                       /       \
									 w3 w4	   w1 w2			
                                     gpu0       gpu1
                                     	   |
                             
                                    每个gpu得到更新后的完整模型
                                           |
                                           gpu0释放w3 w4 g3 g4
                                           gpu1释放w1 w2 g1 g2
                                    	   |
                                下一次forward更新w1 w2 w3 w4
                           
```

##### zero3模型参数分片

```

                           [D1,D2...D_half,Dhalf+1...Dn]
 									|          |
                                    |          |
                                    |          |
                                  batch1     batch2
                                    /           \
              gpu0（model layer-i层的1/2参数a） gpu1（layer-i层的1/2参数b）
                      					 |
                      					 |
                      				   临时共享
                      				    /   \
                      				   b       a
                      				   |       |
                      				gpu0       gpu1
                                    \            /
                                  forward     forward
                                  	|            |
                                  	释放b        释放a
                                  	/            \
                                 backward        backward
                                 	|              |
                               g11 g12 g13 g14   g21 g22 g23 g24
                                 	|              |
                                 	   计算全局梯度
                                    g1=(g11+g21)/2
                                    g2=(g12+g22)/2
                                    g3=(g13+g23)/2
                                    g4=(g14+g24)/2
                                    	  |
                                    reduce + scatter + 通信
                                          |
                                        zero1
                                        /  \      
                                     g4 g3 g2 g1 梯度分片
                                       |    |
                                  	  gpu0	 gpu1  
                                  	   /       \
                                  	                                          		                                  w1=g1+m1/v1*w1   w3=g3+m3/v3*w3                                                                                          
                               w2=g2+m2/v2*w2   w4=g4+m4/v4*w4
                                  		  
                                  	\               /
                                      w1 w2 w3 w4
                                       /       \
									 w3 w4	   w1 w2			
                                     gpu0       gpu1
                                     	   |
                             
                                    每个gpu得到更新后的完整模型
                                           |
                                           gpu0释放w3 w4 g3 g4
                                           gpu1释放w1 w2 g1 g2
                                    	   |
                                下一次forward更新w1 w2 w3 w4
```

