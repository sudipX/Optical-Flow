import numpy as np

def compute_gradients(frame1 : np.ndarray, frame2: np.ndarray) ->tuple:

    f1 = frame1.astype(np.float32)
    f2 = frame2.astype(np.float32)

    I_avg = 0.5 * (f1 + f2) # We are averaging because it reduced temporal aliasing
    # By this we will be calculating gradients from midpoint rather than any endpoints


    Iy,Ix = np.gradient(I_avg) # This calculated central difference
    # f'(x) approximately [f(x+1) - f(x-1)] / 2

    # Iy = row grad, Ix = col grad

    It = f2-f1

    return Ix, Iy, It

H, W = 50, 50
frame1 = np.zeros((H,W), dtype = np.float32)
frame2 = np.zeros((H,W), dtype = np.float32)

frame1[20:30, 15:25] = 1.0 # White square in frame 1
frame2[20:30, 17:27] = 1.0 #same square shifted by 2 pixels in frame 2

Ix, Iy, It = compute_gradients (frame1, frame2)
#Since Ix measures horizontal gradients, it is large are vectical edges
#Since Iy measures vertical gradients, it is large are horizontal edges

print("Max |Ix| :", np.max(np.abs(Ix)))
print("Max |Iy| :", np.max(np.abs(Iy)))
print("Left region It (square departed):", It[20:30,15:17].mean()) #negative
print("Right region It (square arrived):", It[20:30,25:27].mean()) #postive
