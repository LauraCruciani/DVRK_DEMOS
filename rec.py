# coding: utf-8
import os
import cv2
import copy
import numpy as np 
from math import floor
import time
from model import hsm
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from model.submodule import *
from utils.preprocess import get_transform
from utils import *
from matplotlib import pyplot as plt
#from numba import jit
import open3d as o3d
#import sensor_msgs.msg as sensor_msgs
#from pcl.PointCloud import *
#from sensor_msgs.msg import CompressedImage, PointCloud2, PointField
#from models import __models__
import argparse
from core.raft_stereo import RAFTStereo
from core.utils.utils import InputPadder
# from datasets import __datasets__



#--------------- PRELIMINARY OPERATIONS -------------
#---- Load camera parameters ----
def load_camera_param (xml_file):
    #print(xml_file)
    cv_file = cv2.FileStorage(xml_file, cv2.FILE_STORAGE_READ)
    # The intrinsic parameters (intrinsic matrix), respectively for left and right camera
    Kl = cv_file.getNode("M_l").mat()
    Kr = cv_file.getNode("M_r").mat()
    # Focal length
    #focal_length_L = np.array([Kl[0,0], Kl[1,1]])
    #focal_length_R = np.array([Kr[0,0], Kr[1,1]])
    # The distortion parameters, respectively for left and right camera
    dist_l = cv_file.getNode("D_l").mat().flatten()
    dist_r = cv_file.getNode("D_r").mat().flatten()
    # The extrinsic parameters, respectively rotation and translation matrix
    R = cv_file.getNode("R").mat()
    T = cv_file.getNode("T").mat()
    cv_file.release()
    return Kl, Kr, dist_l,dist_r, R, T

#---- HSM loading ----
def HSM_loading(loadmodel, clean, max_disp, level, testres):
    
    os.environ['KMP_DUPLICATE_LIB_OK']='True'

    #cudnn.benchmark = True
    cudnn.benchmark = False
  
    # construct model
    model = hsm(128,clean,level=level)
    model = nn.DataParallel(model, device_ids=[0])
    model.cuda()
    print('loadHSM')
    if loadmodel is not None:
        pretrained_dict = torch.load(loadmodel)
        pretrained_dict['state_dict'] =  {k:v for k,v in pretrained_dict['state_dict'].items() if 'disp' not in k}
        model.load_state_dict(pretrained_dict['state_dict'],strict=False)
    else:
        print('run with random init')
    #print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

    # dry run
    multip = 48
    imgL = np.zeros((1,3,24*multip,32*multip))
    imgR = np.zeros((1,3,24*multip,32*multip))
    imgL = Variable(torch.FloatTensor(imgL).cuda())
    imgR = Variable(torch.FloatTensor(imgR).cuda())
    with torch.no_grad():
        model.eval()
        pred_disp,entropy = model(imgL,imgR)

    model.eval()

    if max_disp>0:
        if max_disp % 16 != 0:
            max_disp = 16 * floor(max_disp/16)
        max_disp = int(max_disp)
    # else:
    #     with open(work_path + '/calib.txt') as f:
    #         lines = f.readlines()
    #         max_disp = int(int(lines[6].split('=')[-1]))

    ## change max disp
    tmpdisp = int(max_disp*testres//64*64)
    if (max_disp*testres/64*64) > tmpdisp:
        model.module.maxdisp = tmpdisp + 64
    else:
        model.module.maxdisp = tmpdisp
    if model.module.maxdisp ==64: model.module.maxdisp=128
    model.module.disp_reg8 =  disparityregression(model.module.maxdisp,16).cuda()
    model.module.disp_reg16 = disparityregression(model.module.maxdisp,16).cuda()
    model.module.disp_reg32 = disparityregression(model.module.maxdisp,32).cuda()
    model.module.disp_reg64 = disparityregression(model.module.maxdisp,64).cuda()
    #print(model.module.maxdisp)

    return model

# def CFNet_loading(loadmodel, max_disp): 
#     cudnn.benchmark = True

#     parser = argparse.ArgumentParser(description='Cascade and Fused Cost Volume for Robust Stereo Matching(CFNet)')
#     parser.add_argument('--model', default='cfnet', help='select a model structure', choices=__models__.keys())
#     parser.add_argument('--maxdisp', type=int, default=192, help='maximum disparity')

#     parser.add_argument('--dataset', default='kitti', help='dataset name', choices=__datasets__.keys())
#     parser.add_argument('--loadckpt', default='./Weights/finetuning_model', help='load the weights from a specific checkpoint')

#     # parse arguments
#     args = parser.parse_args()
#     # model, optimizer
#     model = __models__[args.model](args.maxdisp)
#     #model = nn.DataParallel(model)
#     model.cuda()

#     # load parameters
#     print("loading model {}".format(args.loadckpt))
#     state_dict = torch.load(args.loadckpt)
#     model.load_state_dict(state_dict['model'])
#     model = nn.DataParallel(model)
    
#     # dry run
#     multip = 48
#     imgL = np.zeros((1,3,24*multip,32*multip))
#     imgR = np.zeros((1,3,24*multip,32*multip))
#     imgL = Variable(torch.FloatTensor(imgL).cuda())
#     imgR = Variable(torch.FloatTensor(imgR).cuda())

#     with torch.no_grad():
#         model.eval()
#         outputs = model(imgL, imgR)
#         pred_disp = outputs[-1]
#         pred_disp = tensor2numpy(pred_disp)

#     return model

# def RAFT_loading(loadmodel):
#     DEVICE = 'cuda'
#     parser = argparse.ArgumentParser()
#     #parser.add_argument('--restore_ckpt', help="restore checkpoint", default='./raftstereo-middlebury.pth')
#     parser.add_argument('--save_numpy', action='store_true', help='save output as numpy arrays')
#     parser.add_argument('-l', '--left_imgs', help="path to all first (left) frames", default="datasets/Middlebury/MiddEval3/testH/*/im0.png")
#     parser.add_argument('-r', '--right_imgs', help="path to all second (right) frames", default="datasets/Middlebury/MiddEval3/testH/*/im1.png")
#     parser.add_argument('--output_directory', help="directory to save output", default="demo_output")
#     parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
#     parser.add_argument('--valid_iters', type=int, default=32, help='number of flow-field updates during forward pass')

#     # Architecture choices
#     parser.add_argument('--hidden_dims', nargs='+', type=int, default=[128]*3, help="hidden state and context dimensions")
#     parser.add_argument('--corr_implementation', choices=["reg", "alt", "reg_cuda", "alt_cuda"], default="reg", help="correlation volume implementation")
#     parser.add_argument('--shared_backbone', action='store_true', help="use a single backbone for the context and feature encoders")
#     parser.add_argument('--corr_levels', type=int, default=4, help="number of levels in the correlation pyramid")
#     parser.add_argument('--corr_radius', type=int, default=4, help="width of the correlation pyramid")
#     parser.add_argument('--n_downsample', type=int, default=2, help="resolution of the disparity field (1/2^K)")
#     parser.add_argument('--slow_fast_gru', action='store_true', help="iterate the low-res GRUs more frequently")
#     parser.add_argument('--n_gru_layers', type=int, default=3, help="number of hidden GRU levels")
    
#     args = parser.parse_args()

#     model = torch.nn.DataParallel(RAFTStereo(args), device_ids=[0])
#     model.load_state_dict(torch.load(loadmodel))

#     model = model.module
#     model.to(DEVICE)
#     model.eval()

#     return model

#---- Parameters elaboration ----
def elaboration_param(Kl, Kr, dist_l, dist_r, R, T, h, w, h_res, w_res):
    
    # Computation of matrix Q tha's used for conversion from disparity to depth
    x_ratio = w_res / w
    y_ratio = h_res / h
    [R_l, R_r, P_l, P_r, _] = stereoRectify(Kl, Kr, dist_l, dist_r, h, w, R, T)
    Kl_res = copy.copy(Kl)
    Kr_res = copy.copy(Kr)
    Kl_res[0,:] = Kl_res[0,:] * x_ratio
    Kl_res[1,:] = Kl_res[1,:] * y_ratio
    Kr_res[0,:] = Kr_res[0,:] * x_ratio
    Kr_res[1,:] = Kr_res[1,:] * y_ratio
    [_,_,_,_,Q] = stereoRectify(Kl_res, Kr_res, dist_l, dist_r, h_res, w_res, R, T)  
    
    # Computation of maps used for rectification
    mapx_l, mapy_l = cv2.initUndistortRectifyMap(Kl, dist_l, R_l, P_l, (w,h), cv2.CV_16SC2)    #CV_32F
    mapx_r, mapy_r = cv2.initUndistortRectifyMap(Kr, dist_r, R_r, P_r, (w,h), cv2.CV_16SC2)    #CV_32F

    return Q, mapx_l, mapy_l, mapx_r, mapy_r



#--------------- RECTIFICATION -------------
def stereoRectify(Kl, Kr, dist_l, dist_r, h, w, R, T):
        (R_l, R_r, P_l, P_r, Q, roi1, roi2) = cv2.stereoRectify(cameraMatrix1 = Kl,cameraMatrix2 = Kr,
                                                                distCoeffs1 = dist_l, distCoeffs2 = dist_r,
                                                                imageSize = (h,w),
                                                                R = R, T = T,
                                                                #flags = cv2.CALIB_USE_INTRINSIC_GUESS,
                                                                flags=cv2.CALIB_ZERO_DISPARITY,
                                                                alpha = 0.5,
                                                                newImageSize = (h,w) )
        '''
        cameraMatrix - the two intrinsic matrix
        R – Rotation matrix between the coordinate systems of the first and the second cameras.
        T – Translation vector between coordinate systems of the cameras.
        R1 – Output 3x3 rectification transform (rotation matrix) for the first camera.
        R2 – Output 3x3 rectification transform (rotation matrix) for the second camera.
        P1 – Output 3x4 projection matrix in the new (rectified) coordinate systems for the first camera.
        P2 – Output 3x4 projection matrix in the new (rectified) coordinate systems for the second camera.
        Q – Output 4 \times 4 disparity-to-depth mapping matrix '''
        return R_l, R_r, P_l, P_r, Q

def rectification_opencv(imgL, imgR, Kl, Kr, dist_l, dist_r, P_l, P_r, R_l, R_r, h, w):

    start_rect_time = time.time()

    #Get optimal camera matrix for better undistortion 
    #Kl, roi = cv2.getOptimalNewCameraMatrix(Kl,dist_l,(w,h),0,(w,h))
    #Kr, roi = cv2.getOptimalNewCameraMatrix(Kr,dist_r,(w,h),1,(w,h))

    mapx_l, mapy_l = cv2.initUndistortRectifyMap(Kl, dist_l, R_l, P_l, (w,h), cv2.CV_32F)
    mapx_r, mapy_r = cv2.initUndistortRectifyMap(Kr, dist_r, R_r, P_r, (w,h), cv2.CV_32F)

    # storing the principal points of both cameras
    #(principal_xl, principal_yl) = (int(Kl[0, 2]), int(Kl[1, 2]))
    #(principal_xr, principal_yr) = (int(Kr[0, 2]), int(Kr[1, 2]))

    # undistorting the images witht the calculated undistortion map
    #result_l = np.ndarray(shape=([h, w, 3]), dtype=np.uint8)
    result_l = cv2.remap(imgL, mapx_l, mapy_l, cv2.INTER_LINEAR)

    # undistorting the images witht the calculated undistortion map
    #result_r = np.ndarray(shape=([h, w, 3]), dtype=np.uint8)
    result_r = cv2.remap(imgR, mapx_r, mapy_r, cv2.INTER_LINEAR)

    # cv2.imwrite("rectified_L.png", result_l)
    # cv2.imwrite("rectified_R.png", result_r)

    #plt.subplot(121), plt.imshow(result_l)
    #plt.subplot(122), plt.imshow(result_r)
    #plt.show()

    end_rect_time = time.time()
    rect_time = end_rect_time-start_rect_time
    print ("\nRectification completed in " + str(rect_time))

    return result_l, result_r

def rectification_resize_OLD(img, K, dist, P, R, h, w, h_res, w_res, side, self):

    start_rect_time = time.time()

    #Get optimal camera matrix for better undistortion 
    #K, roi = cv2.getOptimalNewCameraMatrix(K,dist,(w,h),0,(w,h))
    T1 = time.time()
    mapx, mapy = cv2.initUndistortRectifyMap(K, dist, R, P, (w,h), cv2.CV_16SC2)    #CV_32F
    print('T1 ' + str(time.time() - T1))
    # storing the principal points of both cameras
    #(principal_x, principal_y) = (int(K[0, 2]), int(K[1, 2]))

    # undistorting the images witht the calculated undistortion map
    #result = np.ndarray(shape=([h, w, 3]), dtype=np.uint8)
    T2 = time.time()
    result = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)

    print('T2 ' + str(time.time() - T2))

    # Save rectified image
    #cv2.imwrite("rectified_" + str(side) + ".png", result)

    # RESIZE
    img_res = cv2.resize(result, (w_res,h_res), interpolation = cv2.INTER_CUBIC)

    if side == 'L':
        self.imgL_resized = img_res
    else :
        self.imgR_resized = img_res

    # Save resized image
    #cv2.imwrite("img" + str(side) + "_resized.png", img_res)

    print("Total RECT+RES time " + str(side) + ": " + str(time.time() - start_rect_time))

    return img_res

def rectification_resize(img, mapx, mapy, h_res, w_res):
    #start_rect_time = time.time()

    # undistorting the images witht the calculated undistortion map

    result = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)

    # Save rectified image
    #cv2.imwrite("rectified_" + str(side) + ".png", result)
    # cv2.imwrite("./test_final/rec_" + str(side)+"_"+str(count) + ".png", result)
    # RESIZE
    img_res = cv2.resize(result, (w_res,h_res), interpolation = cv2.INTER_CUBIC)

    # Save resized image
    # cv2.imwrite("./test_final/" + str(side) + "_"+str(count)+"_resized.png", img_res)

    #print("Total RECT+RES time " + str(side) + ": " + str(time.time() - start_rect_time))
    return img_res



### PRE-PROCESSING da valutare se inserire 
#(nel caso copiare da new_alg.py)




#--------------- HSM DISPARITY COMPUTATION -------------
def HSM_computation(model, imgL_o, imgR_o, h, w, count):
    # start_disp_time = time.time()
    
    processed = get_transform()

    # resize
    #imgL_o = cv2.resize(imgL_o,None,fx=testres,fy=testres,interpolation=cv2.INTER_CUBIC)
    #imgR_o = cv2.resize(imgR_o,None,fx=testres,fy=testres,interpolation=cv2.INTER_CUBIC)
    imgL = processed(imgL_o).numpy()
    imgR = processed(imgR_o).numpy() 

    imgL = np.reshape(imgL,[1,3,imgL.shape[1],imgL.shape[2]])
    imgR = np.reshape(imgR,[1,3,imgR.shape[1],imgR.shape[2]])

    #imgL_o.shape[:2] = (576,720)
    #imgL.shape[1] = 3
    #imgL.shape[2] = h
    #imgL.shape[3] = w

    ##fast pad
    max_h = int(h // 64 * 64)
    max_w = int(w // 64 * 64)
    if max_h < h: max_h += 64
    if max_w < w: max_w += 64

    top_pad = max_h-h
    left_pad = max_w-w
    imgL = np.lib.pad(imgL,((0,0),(0,0),(top_pad,0),(0,left_pad)),mode='constant',constant_values=0)
    imgR = np.lib.pad(imgR,((0,0),(0,0),(top_pad,0),(0,left_pad)),mode='constant',constant_values=0)

    # test
    imgL = Variable(torch.FloatTensor(imgL).cuda())
    imgR = Variable(torch.FloatTensor(imgR).cuda())


    with torch.no_grad():
        # torch.cuda.synchronize()
        #start_time = time.time()
        pred_disp,_ = model(imgL,imgR)
        # torch.cuda.synchronize()
        #ttime = (time.time() - start_time); #print('time = %.2f' % (ttime*1000) )
    pred_disp = torch.squeeze(pred_disp).data.cpu().numpy()

    
    top_pad   = max_h-imgL_o.shape[0]
    left_pad  = max_w-imgL_o.shape[1]
    #entropy = entropy[top_pad:,:pred_disp.shape[1]-left_pad].cpu().numpy()
    pred_disp = pred_disp[top_pad:,:pred_disp.shape[1]-left_pad]

    # resize to highres
    #pred_disp = cv2.resize(pred_disp/testres,(imgsize[1],imgsize[0]),interpolation=cv2.INTER_LINEAR)

    # clip while keep inf
    invalid = np.logical_or(pred_disp == np.inf,pred_disp!=pred_disp)
    pred_disp[invalid] = np.inf

    # print("disp_max=",np.max(pred_disp[~invalid]))
    # print("disp_min=",np.min(pred_disp))
    #pred_disp = pred_disp/pred_disp[~invalid].max()*255
    #pred_disp = pred_disp/pred_disp[~invalid].max()
    # Save DISPARITY MAP
    #cv2.imwrite('./pic/disp' + str(count) + '.png',pred_disp/pred_disp[~invalid].max()*255)
    # cv2.imwrite('./test_final/disp'+str(count)+'.png',pred_disp/pred_disp[~invalid].max()*255)
    #torch.cuda.empty_cache()
    # cv2.imwrite('./pic_640_128/disp_HSM_'+str(count)+'.png',pred_disp/pred_disp[~invalid].max()*255)
    # disp_time = time.time() - start_disp_time
    # print ("Disparity computation time: " + str(disp_time))
    torch.cuda.empty_cache()
    return pred_disp


def CFNet_computation(model, imgL_o, imgR_o, h, w, count): 
    # start_disp_time = time.time()
    
    processed = get_transform()
    model.eval()

    # resize
    #imgL_o = cv2.resize(imgL_o,None,fx=testres,fy=testres,interpolation=cv2.INTER_CUBIC)
    #imgR_o = cv2.resize(imgR_o,None,fx=testres,fy=testres,interpolation=cv2.INTER_CUBIC)
    imgL = processed(imgL_o).numpy()
    imgR = processed(imgR_o).numpy() 

    imgL = np.reshape(imgL,[1,3,imgL.shape[1],imgL.shape[2]])
    imgR = np.reshape(imgR,[1,3,imgR.shape[1],imgR.shape[2]])

    #imgL_o.shape[:2] = (576,720)
    #imgL.shape[1] = 3
    #imgL.shape[2] = h
    #imgL.shape[3] = w
    max_h = int(imgL.shape[2] // 64 * 64)
    max_w = int(imgL.shape[3] // 64 * 64)
    if max_h < imgL.shape[2]: max_h += 64
    if max_w < imgL.shape[3]: max_w += 64


    top_pad = max_h-imgL.shape[2]
    left_pad = max_w-imgL.shape[3]
    imgL = np.lib.pad(imgL,((0,0),(0,0),(top_pad,0),(0,left_pad)),mode='constant',constant_values=0)
    imgR = np.lib.pad(imgR,((0,0),(0,0),(top_pad,0),(0,left_pad)),mode='constant',constant_values=0)

    # test
    imgL = Variable(torch.FloatTensor(imgL).cuda())
    imgR = Variable(torch.FloatTensor(imgR).cuda())


    with torch.no_grad():
        torch.cuda.synchronize()
        #start_time = time.time()
        model.eval()
        outputs = model(imgL, imgR)
        pred_disp = outputs[-1]
        #pred_disp = tensor2numpy(pred_disp)

        torch.cuda.synchronize()
        #ttime = (time.time() - start_time); print('time = %.2f' % (ttime) )
    pred_disp = torch.squeeze(pred_disp).data.cpu().numpy()

    top_pad   = max_h-imgL_o.shape[0]
    left_pad  = max_w-imgL_o.shape[1]
    #entropy = entropy[top_pad:,:pred_disp.shape[1]-left_pad].cpu().numpy()
    pred_disp = pred_disp[top_pad:,:pred_disp.shape[1]-left_pad]
    imgsize = imgL_o.shape[:2]

    # resize to highres
    pred_disp = cv2.resize(pred_disp,(imgsize[1],imgsize[0]),interpolation=cv2.INTER_LINEAR)

    # clip while keep inf
    invalid = np.logical_or(pred_disp == np.inf,pred_disp!=pred_disp)
    pred_disp[invalid] = np.inf

    # print("disp_max=",np.max(pred_disp[~invalid]))
    # print("disp_min=",np.min(pred_disp))
    #pred_disp = pred_disp/pred_disp[~invalid].max()*255
    #pred_disp = pred_disp/pred_disp[~invalid].max()
    # Save DISPARITY MAP
    cv2.imwrite('./pic_CFNet/disp'+str(count)+'.png',pred_disp/pred_disp[~invalid].max()*255)
    # cv2.imwrite('./test_final/disp'+str(count)+'.png',pred_disp/pred_disp[~invalid].max()*255)
    #torch.cuda.empty_cache()
    # cv2.imwrite('./pic_640_128/disp_HSM_'+str(count)+'.png',pred_disp/pred_disp[~invalid].max()*255)
    # disp_time = time.time() - start_disp_time
    # print ("Disparity computation time: " + str(disp_time))
    torch.cuda.empty_cache()
    return pred_disp


def RAFT_computation(model, imgL_o, imgR_o):
    model.eval()
    with torch.no_grad():
        padder = InputPadder(imgL_o.shape, divis_by=32)
        imgL, imgR = padder.pad(imgL_o, imgR_o)
        _, flow_up = model(imgL, imgR, iters=32, test_mode=True)
        pred_disp= flow_up.cpu().numpy().squeeze()
    
    return pred_disp



#--------------- 3D POINT CLOUD COMPUTATION -------------
def point_cloud(dispTo3D, colors, Q, h, w) :
    # start_cloud_time = time.time()

    #-------------- POINT CLOUD COMPUTATION ------------------
    points_3D = cv2.reprojectImageTo3D(dispTo3D, Q)
    # I resize the point cloud by factor 3 to allign it to the ground truth
    #points_3D = points_3D * 3
    points_3D = np.reshape(points_3D,(h*w,3))

    #-------------- INVALID POINTS ELIMINATION ------------------                  
    # Infinite coordinates elimination
    points_3D[~np.isfinite(points_3D)] = 0
    # Black points elimination
    #points_3D[np.all(colors < 32, axis = 1)] = [0,0,0]

    # cloud_time = time.time()-start_cloud_time
    # print ("Point cloud completed in " + str(cloud_time))

    return points_3D


#--------------- POINT CLOUD VISUALIZATION -------------
# Create ply file
def ply_creation(colors, points_3D_hsm) :
    colors = colors/255     # Because open3d works with color's values [0,1] instead of [0,255]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_3D_hsm)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    return pcd
    
# Conversion to PointCloud2 ros mesg
# https://github.com/spillai/pybot/blob/df6989d90860e88de42a3183d3af6c4c1c06336a/pybot/externals/ros/pointclouds.py
def xyzrgb_array_to_pointcloud2(points, colors, frame_id, stamp=None,  seq=None):
    #Create a sensor_msgs.PointCloud2 from an array of points.
    msg = PointCloud2()
    assert(points.shape == colors.shape)

    buf = []

    if stamp:
        msg.header.stamp = stamp
    if frame_id:
        msg.header.frame_id = frame_id
    if seq: 
        msg.header.seq = seq
    if len(points.shape) == 3:
        msg.height = points.shape[1]
        msg.width = points.shape[0]
    else:
        N = len(points)
        xyzrgb = np.array(np.hstack([points, colors]), dtype=np.float32)
        msg.height = 1
        msg.width = N

    msg.fields = [
        PointField('x', 0, PointField.FLOAT32, 1),
        PointField('y', 4, PointField.FLOAT32, 1),
        PointField('z', 8, PointField.FLOAT32, 1),
        PointField('r', 12, PointField.FLOAT32, 1),
        PointField('g', 16, PointField.FLOAT32, 1),
        PointField('b', 20, PointField.FLOAT32, 1)
    ]
    msg.is_bigendian = False
    msg.point_step = 24
    msg.row_step = msg.point_step * N
    msg.is_dense = True; 
    msg.data = xyzrgb.tostring()

    return msg
        





'''
pcd = o3d.io.read_point_cloud(work_path + "/HSM_point.ply")
o3d.visualization.draw_geometries([pcd],
                                  zoom=0.3412,
                                  front=[0.4257, -0.2125, -0.8795],
                                  lookat=[2.6172, 2.0475, 1.532],
                                  up=[-0.0694, -0.9768, 0.2024])
'''

#-------------- GROUND TRUTH ------------------
'''
colors = cv2.cvtColor(imgL, cv2.COLOR_BGR2RGB)                   
depth_path = work_path + '/gt.txt'    
ground_truth = np.loadtxt(depth_path)
#Create ply file to visualize ground truth
[h,w] = size
cc = np.reshape(colors,(h*w,3))
create_output(ground_truth, cc, work_path + '/gt.ply')
'''

def create_output(vertices, colors, filename):
    colors = colors.reshape(-1,3)
    vertices = np.hstack([vertices.reshape(-1,3),colors])

    ply_header = '''ply
        format ascii 1.0
        element vertex %(vert_num)d
        property float x
        property float y
        property float z
        property uchar red
        property uchar green
        property uchar blue
        end_header
        '''
    with open(filename, 'w') as f:
        f.write(ply_header %dict(vert_num=len(vertices)))
        np.savetxt(f,vertices,'%f %f %f %d %d %d')
