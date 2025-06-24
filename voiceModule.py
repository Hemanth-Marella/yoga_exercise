
import platform
import pyttsx3
import os
import threading
    

class VoicePlay():
    
    def __init__(self):
        super().__init__()
        self.system = platform.system()
        self.engine = None
        self.callBack = None
        self.opId = -1
        self.playingText = []
        self.isAudio_Playing_Completed = True
        self.isVoicePlaying = False
        if self.system == 'Windows':
            try:
                self.engine = pyttsx3.init()
                self.configureEngine(self.engine)
            except Exception as e:
                pass
                # print(f"Error initializing TTS engine on Windows: {e}")

    def setID(self, id):
        self.exerciseID = id
    
    def configureEngine(self, engine):
    
        try:
            voices = engine.getProperty('voices')
            for voice in voices:
                if 'en_US' in voice.languages or 'English' in voice.name:
                    if 'female' not in voice.name.lower() and 'zira' not in voice.name.lower():
                        engine.setProperty('voice', voice.id)
                        break
            # Remove or comment out the next line to use default voice speed
            engine.setProperty('rate', 190)
        except KeyError as e:
            pass
        except Exception as e:
            pass

    def playAudio(self,textToPlay, callBack=None,  opId=None, play=False ):

        if not textToPlay:
            return
        
        # self.stopVoice()

        self.isVoicePlaying = True
        self.isAudio_Playing_Completed = False
        self.playingText = textToPlay
        self.callBack = callBack
        self.play=play
        self.opId = opId
        self.playVoiceBackground(play=self.play) 


    def playVoice(self, play):
        if not self.playingText:
            # print("Error: playingText is empty. No text to play.")
            return

        if self.system == 'Darwin':  # macOS
            try:
                self.length = 0
                if play:
                    for text_to_play in self.playingText :
                        self.length+=1
                        os.system(f'say "{text_to_play}"')
                          
                if self.callBack:
                    self.isVoicePlaying = False
                    self.isAudio_Playing_Completed = True
                    self.callBack(self.opId)
                else:
                    self.isVoicePlaying = False
                    self.isAudio_Playing_Completed = True
                    return
                
            except Exception as e:
                print(f"Error using system 'say' command on macOS: {e}")
        
        elif self.system == 'Windows' and self.engine is not None:  # Windows
            try:
                self.length = 0
                if play:
                    for text_to_play in self.playingText :
                        self.length+=1
                        self.engine.say(text_to_play)
                        self.engine.runAndWait()
                
                if self.callBack:
                    self.isVoicePlaying = False
                    self.isAudio_Playing_Completed = True
                    self.callBack(self.opId)
                else:
                    self.isVoicePlaying = False
                    self.isAudio_Playing_Completed = True
                    return
                    
            except Exception as e:
                pass
                # print(f"Error during speech playback on Windows: {e}")
        
        elif self.system == 'Linux':  # Linux
            try:
                self.length = 0
                if play:
                    for text_to_play in self.playingText :
                        self.length+=1
                        self.isAudio_Playing_Completed = False
                        os.system(f'espeak "{text_to_play}"')
                        
                self.isAudio_Playing_Completed = True
                if self.callBack:
                    self.callBack(self.opId)
                else:
                    return
            except Exception as e:
                pass
                # print(f"Error using system 'espeak' command on Linux: {e}")
        else:
            pass
            # print(f"Unsupported operating system: {self.system}")

    def playVoiceBackground(self, play):
        """Plays the voice in the background asynchronously."""
        threading.Thread(target = self.playVoice, 
                         daemon = True,
                         args = (play,)
                        ).start()
        
    def stopVoice(self):
        """Stop currently speaking voice."""
        if self.system == 'Windows' and self.engine:
            try:
                self.engine.stop()
            except Exception:
                pass
        self.isVoicePlaying = False
        self.isAudio_Playing_Completed = True


    def startEngine(self):
        try:
            
            self.engine = pyttsx3.init()
            self.configureEngine(self.engine)
            return True
        except Exception as e:
            pass
            # print(f"Error initializing TTS engine on Windows: {e}")

# if __name__ == '__main__':
#     obj = VoicePlay()
#     obj.playAudio(["hello hi Bhujangasana, or Cobra Pose, is a yoga posture that opens the heart,"],play=True)
