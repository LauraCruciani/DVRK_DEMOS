#!/usr/bin/env python
import rospy
import numpy as np
import cv2
from sensor_msgs.msg import CompressedImage

class ImageSubscriber:
    def __init__(self):
        # Inizializza il nodo ROS
        rospy.init_node('image_subscriber', anonymous=True)
        
        # Topic (in questo caso immagine dx e sx)
        self.sub_left = rospy.Subscriber("/endoscope/raw/left/image_raw/compressed", CompressedImage, self.callback_left, queue_size=1)
        self.sub_right = rospy.Subscriber("/endoscope/raw/right/image_raw/compressed", CompressedImage, self.callback_right, queue_size=1)
        
        rospy.loginfo("Sottoscrizione ai topic completata. In attesa di immagini...")

    def callback_left(self, ros_data):
        """ Callback per il topic dell'immagine sinistra """
        np_arr = np.frombuffer(ros_data.data, np.uint8)  # Conversione da ROS a OpenCV
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        #rospy.loginfo("Ricevuta immagine sinistra")
        cv2.imshow("Left Image", img)
        cv2.waitKey(1) ## Aggiorna la finestra

    def callback_right(self, ros_data):
        """ Callback per il topic dell'immagine destra """
        np_arr = np.frombuffer(ros_data.data, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        rospy.loginfo("Ricevuta immagine destra")
        cv2.imshow("Right Image", img)
        cv2.waitKey(1)

if __name__ == '__main__':
    try:
        subscriber = ImageSubscriber()
        while not rospy.is_shutdown():
            cv2.waitKey(1)  # Mantiene la finestra OpenCV aperta
        rospy.spin()  # Mantiene il nodo in esecuzione
    except rospy.ROSInterruptException:
        rospy.loginfo("Nodo interrotto.")
    finally:
        cv2.destroyAllWindows()
