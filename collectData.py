import os
import cv2

cap = cv2.VideoCapture(0)
directory = 'Image/'

actions = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
special_actions = {'1': 'thankyou', '2': 'iloveyou', '3': 'hello'}

# Create directories if they do not exist
for action in actions:
    os.makedirs(os.path.join(directory, action), exist_ok=True)
for special_action in special_actions.values():
    os.makedirs(os.path.join(directory, special_action), exist_ok=True)

while True:
    _, frame = cap.read()
    count = {action: len(os.listdir(directory + "/" + action)) for action in actions}
    special_count = {key: len(os.listdir(directory + "/" + value)) for key, value in special_actions.items()}

    row = frame.shape[1]
    col = frame.shape[0]
    cv2.rectangle(frame, (0, 40), (300, 400), (255, 255, 255), 2)
    cv2.imshow("data", frame)
    cv2.imshow("ROI", frame[40:400, 0:300])
    frame = frame[40:400, 0:300]
    interrupt = cv2.waitKey(10)
    
    for action in actions:
        if interrupt & 0xFF == ord(action.lower()):
            cv2.imwrite(directory + action + '/' + str(count[action]) + '.png', frame)

    if interrupt & 0xFF == ord('1'):  # Key '1'
        action = special_actions['1']
        cv2.imwrite(directory + action + '/' + str(special_count['1']) + '.png', frame)
    elif interrupt & 0xFF == ord('2'):  # Key '2'
        action = special_actions['2']
        cv2.imwrite(directory + action + '/' + str(special_count['2']) + '.png', frame)
    elif interrupt & 0xFF == ord('3'):  # Key '3'
        action = special_actions['3']
        cv2.imwrite(directory + action + '/' + str(special_count['3']) + '.png', frame)

    if interrupt & 0xFF == 27:  # Press 'Esc' to exit the loop
        break

cap.release()
cv2.destroyAllWindows()
