import numpy as np
import keyboard
import cv2

print(cv2.__version__)

def mouse_draw(event, x,y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        raudoni.append((x,y))
    if event == cv2.EVENT_RBUTTONDOWN:
        melyni.append((x,y))
    if event == cv2.EVENT_MOUSEWHEEL:
    #if keyboard.is_pressed('e'):
        zali.append((x,y))


raudoni = []
melyni = []
zali = []
laukas = np.ones((500,500,3),dtype=np.uint8)*255
pav = 'langelis'
cv2.namedWindow(pav)
cv2.setMouseCallback(pav,mouse_draw)

while True:
    for i in range(0,len(raudoni)):
        cv2.circle(laukas,(raudoni[i][0],raudoni[i][1]),10,(0,0,255),-1)
    for i in range(0,len(melyni)):
        cv2.circle(laukas,(melyni[i][0],melyni[i][1]),10,(255,0,0),-1)
    for i in range(0,len(zali)):
        cv2.circle(laukas,(zali[i][0],zali[i][1]),10,(0,255,0),-1)
    
    cv2.imshow(pav,laukas)
    #to break while loop bite push the esc button
    if cv2.waitKey(10) == 27:
        break
cv2.destroyAllWindows()


#ANN modelis MLP - multy layer perceptron
model = cv2.ml.ANN_MLP_create()
layers = np.int32([2, 100, 100, 3])

model.setLayerSizes(layers)
model.setTrainMethod(cv2.ml.ANN_MLP_BACKPROP)
model.setBackpropMomentumScale(0.0)
model.setBackpropWeightScale(0.001)
model.setTermCriteria((cv2.TERM_CRITERIA_COUNT, 100, 0.01))
model.setActivationFunction(cv2.ml.ANN_MLP_SIGMOID_SYM, 2, 1)

mokymo_imtis = np.zeros((len(raudoni)+len(melyni)+len(zali), 2), dtype=np.float32)
target = np.zeros((len(raudoni)+len(melyni)+len(zali), 3), dtype=np.float32)

for i in range(0,len(raudoni)):
    for j in range(0,2):
        mokymo_imtis[i,j] = raudoni[i][j]
    target[i] = [1.0,0.0,0.0]

for i in range(0,len(melyni)):
    for j in range(0,2):
        mokymo_imtis[i + len(raudoni),j] = melyni[i][j]
    target[i + len(raudoni)] = [0.0,1.0,0.0]

for i in range(0,len(zali)):
    for j in range(0,2):
        mokymo_imtis[i +len(raudoni)+ len(melyni),j] = zali[i][j]
    target[i + len(melyni)+len(raudoni)] = [0.0,0.0,1.0]

print(mokymo_imtis)
print(target)

model.train(mokymo_imtis,cv2.ml.ROW_SAMPLE,target)

sample = np.array([[50, 100]],dtype=np.float32)

laukas = np.ones((500,500,3),dtype=np.uint8)*255

for i in range(0,500):
    for j in range(0,500):
        out = model.predict(np.float32([[i,j]]))
        out = out[1][0]
        result = np.where(out == np.amax(out))
        result = result[0]
        #print(result[0])
        if result[0] == 0:
            cv2.circle(laukas,(i,j),1,(210,210,255),-1)
        elif result[0] == 1:
            cv2.circle(laukas,(i,j),1,(210,210,100),-1)
        else:
            cv2.circle(laukas,(i,j),1,(210,100,255),-1)

for i in range(0,len(raudoni)):
    cv2.circle(laukas,(raudoni[i][0],raudoni[i][1]),10,(0,0,255),-1)
for i in range(0,len(melyni)):
    cv2.circle(laukas,(melyni[i][0],melyni[i][1]),10,(255,0,0),-1)
for i in range(0,len(zali)):
    cv2.circle(laukas,(zali[i][0],zali[i][1]),10,(0,255,0),-1)

cv2.imshow(pav,laukas)
cv2.waitKey()
cv2.destroyAllWindows()
     #   

print('geras')