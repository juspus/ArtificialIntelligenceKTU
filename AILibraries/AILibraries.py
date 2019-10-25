import cv2
import numpy as np



def mouse_draw(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        circles.append([x, y])
    if event == cv2.EVENT_FLAG_RBUTTON:
        squares.append([x, y])

area = np.ones((300, 300, 3), np.uint8) *255

pav = 'Langelis'
cv2.namedWindow(pav)

cv2.setMouseCallback(pav, mouse_draw);

circles = []
squares = []

while True:
    for circle in circles:
        cv2.circle(area, (circle[0], circle[1]), 3, (0,0,255))
        
    for square in squares:
        cv2.rectangle(area, (square[0]-3, square[1]-3), (square[0]+3, square[1]+3), (255,0,0))
    
    cv2.imshow(pav, area)
    if cv2.waitKey(10) == 27:
        break


model = cv2.ml.ANN_MLP_create()
layer_size = np.int32([2, 20, 1])
model.setLayerSizes(layer_size)
model.setTrainMethod(cv2.ml.ANN_MLP_BACKPROP)
model.setBackpropMomentumScale(0.0)
model.setBackpropWeightScale(0.001)
model.setTermCriteria((cv2.TERM_CRITERIA_COUNT, 100, 0.01))
model.setActivationFunction(cv2.ml.ANN_MLP_SIGMOID_SYM, 2, 1)

mokymo_imtis = np.zeros((len(circles)+len(squares),2))
tikslo_vektorius = np.zeros((len(circles)+len(squares), 1))

for i in range(0, len(circles)):
    for j in range(0,2):
        mokymo_imtis[i, j] = circles[i][j]
    tikslo_vektorius[i]=0

for i in range(0, len(squares)):
    for j in range(0,2):
        mokymo_imtis[i, j] = squares[i][j]
    tikslo_vektorius[i]=1

model.train(np.float32(mokymo_imtis), cv2.ml.ROW_SAMPLE, np.float32(tikslo_vektorius))

sample = np.float32(np.array([20,10]))
print(model.predict(sample))
#print(model)