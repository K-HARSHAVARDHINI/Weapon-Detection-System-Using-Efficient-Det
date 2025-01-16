import cv2
import numpy as np
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
from datetime import datetime
import os

# Email setup
from_email = 'kani082003kj@gmail.com'  # Replace with your Gmail address
app_password = 'gvny ietz warz kgtd'  # Use your 16-digit app password here
to_emails = ['2021ad0588@svce.ac.in']  # Add multiple email addresses

# List of classes to detect (Knife)
classes = ["Knife"]


net = cv2.dnn.readNet("effdet_training_2000.weights", "effdet_testing.cfg")


output_layer_names = net.getUnconnectedOutLayersNames()

# Assign random colors to the classes (Knife)
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# Function to send email alert with timestamp and attached image
def send_email(subject, body, attachment_path):
    # Set up the MIME
    msg = MIMEMultipart()
    msg['From'] = from_email
    msg['To'] = ", ".join(to_emails)  # Join all recipient emails with commas
    msg['Subject'] = subject
    
    # Add body to the email
    msg.attach(MIMEText(body, 'plain'))

    # Attach the image file
    with open(attachment_path, 'rb') as f:
        mime_base = MIMEBase('application', 'octet-stream')
        mime_base.set_payload(f.read())
        encoders.encode_base64(mime_base)
        mime_base.add_header('Content-Disposition', f'attachment; filename="{os.path.basename(attachment_path)}"')
        msg.attach(mime_base)

    try:
        # Set up the server and send the email
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(from_email, app_password)
        text = msg.as_string()
        server.sendmail(from_email, to_emails, text)
        server.quit()
        print("Email with attachment sent successfully!")
    except Exception as e:
        print(f"Failed to send email: {e}")

# Enter file name for video, or press Enter to use the webcam
def value():
    val = input("Enter file name or press enter to open camera: \n")
    if val == "":
        val = 0  # This will use the webcam if no video is provided
    return val

# Open video or webcam feed
cap = cv2.VideoCapture(value())

alert_sent = False  # Variable to track if alert has been sent

while True:
    # Read the next frame from the video or webcam
    ret, img = cap.read()
    if not ret:
        print("Error: Failed to read a frame from the video source.")
        break

    height, width, channels = img.shape


    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layer_names)

    # Initialize lists for detected class IDs, confidences, and bounding boxes
    class_ids = []
    confidences = []
    boxes = []

    # Process the outputs from the network
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            # If the confidence is above the threshold, consider it a detection
            if confidence > 0.5:
                # Get the coordinates and size of the bounding box
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Calculate top-left corner of the bounding box
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                # Add bounding box, confidence, and class ID to the lists
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Perform Non-Maximum Suppression (NMS) to avoid multiple detections for the same object
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    if len(indexes) == 0:
        # No weapon detected
        print("No weapon detected.")
    else:
        font = cv2.FONT_HERSHEY_PLAIN
        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                confidence_score = confidences[i]

                # Print confidence score (accuracy) in the command prompt
                print(f"{classes[class_ids[i]]} detected with {confidence_score * 100:.2f}% confidence")

                label = f"{classes[class_ids[i]]} {confidence_score:.2f}"
                color = colors[class_ids[i]]
                cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                cv2.putText(img, label, (x, y + 30), font, 3, color, 3)

                # Send alert only once per detection instance
                if not alert_sent:
                    # Save the current frame with bounding boxes as an image
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    img_path = f"weapon_detection_{timestamp}.jpg"
                    cv2.imwrite(img_path, img)

                    # Set up email details
                    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    subject = f"Weapon Detection Alert: {classes[class_ids[i]]} detected"
                    body = f"Alert: {classes[class_ids[i]]} detected with {confidence_score * 100:.2f}% confidence.\nTimestamp: {timestamp}"

                    # Send email alert with the image attached
                    send_email(subject, body, img_path)
                    print("Email alert with image sent.")

                    alert_sent = True  # Prevent multiple alerts for the same detection

                    # Remove the image file after sending the email
                    os.remove(img_path)

    # Display the processed image with bounding boxes
    cv2.imshow("Weapon Detection", img)
    key = cv2.waitKey(1)

    # Exit the loop if the user presses the 'Esc' key
    if key == 27:
        break
    elif alert_sent:  # Reset alert flag after detection clears
        alert_sent = False

# Release the video capture object and close any OpenCV windows
cap.release()
cv2.destroyAllWindows()
