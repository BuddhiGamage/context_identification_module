from connection import Connection
import qi
from playsound import playsound
import os
import time

#creating the connection
pepper = Connection()
session = pepper.connect('10.0.0.244', '9559')

# Create a proxy to the AL services
behavior_mng_service = session.service("ALBehaviorManager")
tts_service = session.service("ALTextToSpeech")

# Play an animation
def animation(state,prompt):

    behavior_mng_service .stopAllBehaviors()
    behavior_mng_service .startBehavior("pkg/"+state)
    tts_service.say(prompt)

