import cv2

def read_image():
   
    img = cv2.imread("resources/lena.png")  # Read an image

    cv2.imshow("Lena", img)  # Display the image in a window
    # Used to add a delay to see the image in ms.
    # delay = 0 means wait indefinitely until a key is pressed
    # delay = 1000 means wait for 1 second before closing the window
    cv2.waitKey(1000)  


def read_video():
    frame_width = 5000
    frame_height = 1080
    # Open the video file or 0 for webcam
    cap = cv2.VideoCapture(0)
    # cap.set(3, frame_width) 
    # cap.set(4, frame_height)  
    # Check if the video opened successfully
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    while True:
        # Read a frame from the video
        success, frame = cap.read()
        # helps to resize the frame to any resolution we want
        cv2.resize(frame, (frame_width, frame_height))

        # If the frame was not grabbed, we have reached the end of the video
        if not success:
            break

        # Display the frame in a window
        cv2.imshow("Video", frame)

        # Wait for 1 ms and check if 'q' key is pressed to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture object and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()


def main():
    #read_image()  # Call the function to read and display an image
    read_video()  # Call the function to read and display a video

if __name__ == "__main__":
    main()  # Call the main function to start the program