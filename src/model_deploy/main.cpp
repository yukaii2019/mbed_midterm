#include "accelerometer_handler.h"
#include "config.h"
#include "magic_wand_model_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/kernels/micro_ops.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"
#include "uLCD_4DGL.h"
#include "DA7212.h"

#define MODE_RING 0
#define MODE_SLOPE 1
#define MODE_ONE 2
#define MODE_NON 3
DA7212 audio;
Serial pc(USBTX, USBRX);
uLCD_4DGL uLCD(D1, D0, D2);

int PredictGesture(float* output) {
  static int continuous_count = 0;
  static int last_predict = -1;
  int this_predict = -1;
  for (int i = 0; i < label_num; i++) {
    if (output[i] > 0.8) this_predict = i;
  }
  if (this_predict == -1) {
    continuous_count = 0;
    last_predict = label_num;
    return label_num;
  }
  if (last_predict == this_predict) {
    continuous_count += 1;
  } else {
    continuous_count = 0;
  }
  last_predict = this_predict;
  if (continuous_count < config.consecutiveInferenceThresholds[this_predict]) {
    return label_num;
  }
  continuous_count = 0;
  last_predict = -1;
  return this_predict;
}
Thread t(osPriorityNormal);
Thread t2(osPriorityNormal);
Thread t3(osPriorityNormal);
Thread t4(osPriorityNormal);
Thread t5(osPriorityNormal);
//Thread t6(osPriorityNormal);


EventQueue queue1(32*EVENTS_EVENT_SIZE);
EventQueue queue2(32*EVENTS_EVENT_SIZE);
EventQueue queue3(32*EVENTS_EVENT_SIZE);
EventQueue queue4(32*EVENTS_EVENT_SIZE);
EventQueue queue5(32*EVENTS_EVENT_SIZE);
//EventQueue queue6(32*EVENTS_EVENT_SIZE);

InterruptIn sw2(SW2);
InterruptIn sw3(SW3);
DigitalOut greenled(LED2);
int** song;
int** note_len;
char** song_name;
char serialInBuffer[32];
int serialCount;
int load_song_num = 0;
int* song_len;

int mode = MODE_NON;

bool forwar = false;
bool backward = false;



int16_t waveform[kAudioTxBufferSize];
int idC;
bool song_playing_flg = false;

int song_index_playing = 0;
int tmp_song_index_playing = 0;
bool cut_song = false;

bool now_loadong_song_flg = false;



constexpr int kTensorArenaSize = 60 * 1024;
uint8_t tensor_arena[kTensorArenaSize];
static tflite::MicroErrorReporter micro_error_reporter;
tflite::ErrorReporter* error_reporter;
const tflite::Model* model;
static tflite::MicroOpResolver<6> micro_op_resolver;
tflite::MicroInterpreter* interpreter;
TfLiteTensor* model_input;
int input_length;
TfLiteStatus setup_status;

void initial(){
  error_reporter = &micro_error_reporter;
  model = tflite::GetModel(g_magic_wand_model_data);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    error_reporter->Report(
        "Model provided is schema version %d not equal "
        "to supported version %d.",
        model->version(), TFLITE_SCHEMA_VERSION);
    //return -1;
  }
  micro_op_resolver.AddBuiltin(
      tflite::BuiltinOperator_DEPTHWISE_CONV_2D,
      tflite::ops::micro::Register_DEPTHWISE_CONV_2D());
  micro_op_resolver.AddBuiltin(tflite::BuiltinOperator_MAX_POOL_2D,
                               tflite::ops::micro::Register_MAX_POOL_2D());
  micro_op_resolver.AddBuiltin(tflite::BuiltinOperator_CONV_2D,
                               tflite::ops::micro::Register_CONV_2D());
  micro_op_resolver.AddBuiltin(tflite::BuiltinOperator_FULLY_CONNECTED,
                               tflite::ops::micro::Register_FULLY_CONNECTED());
  micro_op_resolver.AddBuiltin(tflite::BuiltinOperator_SOFTMAX,
                               tflite::ops::micro::Register_SOFTMAX());
  micro_op_resolver.AddBuiltin(tflite::BuiltinOperator_RESHAPE,
                             tflite::ops::micro::Register_RESHAPE(), 1);
  // Build an interpreter to run the model with

  static tflite::MicroInterpreter static_interpreter(
      model, micro_op_resolver, tensor_arena, kTensorArenaSize, error_reporter);
  interpreter = &static_interpreter;
  // Allocate memory from the tensor_arena for the model's tensors
  interpreter->AllocateTensors();
  model_input = interpreter->input(0);
  if ((model_input->dims->size != 4) || (model_input->dims->data[0] != 1) ||
      (model_input->dims->data[1] != config.seq_length) ||
      (model_input->dims->data[2] != kChannelNumber) ||
      (model_input->type != kTfLiteFloat32)) {
    error_reporter->Report("Bad input tensor parameters in model");
    //return -1;
  }
  input_length = model_input->bytes / sizeof(float);
  setup_status = SetupAccelerometer(error_reporter);
  if (setup_status != kTfLiteOk) {
    error_reporter->Report("Set up failed\n");
    //return -1;
  }
  error_reporter->Report("Set up successful...\n");
}
void playNote(int freq){
    for(int i=0;i<kAudioTxBufferSize;i++){
        waveform[i] = (int16_t) (sin((double)i * 2. * M_PI / (double) (kAudioSampleFrequency / freq)) * ((1<<16) - 1));
    }
    for(int j=0;j<kAudioSampleFrequency / kAudioTxBufferSize;++j){
        audio.spk.play(waveform,kAudioTxBufferSize);
    }
}
void play_song(int index){
  song_playing_flg = true;
  for(int i=0;i<song_len[index];i++){
        idC=queue4.call_every(1,playNote,song[index][i]);
        if(song[index][i] == song[index][i+1]){
            wait(note_len[index][i]-0.1);
        }
        else {
            wait(note_len[index][i]);
        }
        int a=queue4.cancel(idC);
        if(song[index][i] == song[index][i+1]){
            wait(0.1);
        }
        pc.printf("%d\r\n",a);
        //if(cut_song == true)break;
  }
  song_playing_flg = false;
} /*
void play_song_thread(){
  int stop;
    //if(now_loadong_song_flg == false){
      if(load_song_num!=0&&song_playing_flg == false){
        //stop = queue5.call(play_song,song_index_playing);
        play_song(song_index_playing);
        if(song_index_playing == load_song_num-1){
          song_index_playing = 0;
        }
        else{
          song_index_playing++;
        }
      }  
 //}
}*/
void load_song(int index,int len){
  int i=0;
  serialCount = 0;
  while(i<len){
    if(pc.readable()){
      serialInBuffer[serialCount] = pc.getc();
      serialCount++;
      if(serialCount == 4){
        serialInBuffer [serialCount] = '\0';
        song[index][i] = (int)atof(serialInBuffer);
        serialCount = 0;
        i++;
      }
    }
  }
  i = 0;
  while(i<len){
    if(pc.readable()){
      serialInBuffer[serialCount] = pc.getc();
      serialCount++;
      if(serialCount == 4){
        serialInBuffer [serialCount] = '\0';
        note_len[index][i] = (int)atof(serialInBuffer);
        serialCount = 0;
        i++;
      }
    }
  }
}
void load_song_2(){
  serialCount = 0;
  greenled = 0;
  while(1){
    if(pc.readable()){
      serialInBuffer[serialCount] = pc.getc();
      serialCount++;
      if(serialCount == 4){
        serialInBuffer [serialCount] = '\0';
        load_song_num = (int)atof(serialInBuffer);
        serialCount = 0;
        break;
      }
    }
  }
  song_len = new int[load_song_num];
  for (int i=0;i<load_song_num;i++){
    while(1){
      if(pc.readable()){
        serialInBuffer[serialCount] = pc.getc();
        serialCount++;
        if(serialCount == 4){
          serialInBuffer [serialCount] = '\0';
          song_len[i] = (int)atof(serialInBuffer);
          serialCount = 0;
          break;
        }
      }
    }
  }
  song = new int* [load_song_num];
  for(int i=0;i<load_song_num;i++){
    song[i] = new int[song_len[i]];
  }
  note_len = new int* [load_song_num];
  for(int i=0;i<load_song_num;i++){
    note_len[i] = new int[song_len[i]];
  }
  for(int i=0;i<load_song_num;i++){
    load_song(i,song_len[i]);
  }
  greenled = 1;
}
void load_song_name(void){
  greenled = 0;
  int i=0;
  int j=0;
  song_name = new char*[load_song_num];
  for(int k=0;k<load_song_num;k++){
    song_name[k] = new char[50];
  }
  while(1){
    if(pc.readable()){
      char c_get;
      c_get = (char)pc.getc();
      //pc.printf("aa ");
      //pc.printf("%c ",c_get);
      if(c_get!='#'){
        song_name[i][j++] = c_get;
      }
      else{
        song_name[i][j] = '\0';
        j = 0;
        i++;
        if(i==load_song_num){
          break;
        }  
      }
    }
  }
  
  greenled = 1;
}
void unload_song(){
  if(load_song_num!=0){
    for(int i=0;i<load_song_num;i++){
      delete[] song[i];
    }
    delete [] song;
  
    for(int i=0;i<load_song_num;i++){
      delete[] note_len[i];
    }
    delete [] note_len;
  
    for(int i=0;i<load_song_num;i++){
      delete[] song_name[i];
    }
    delete [] song_name;
  
    delete [] song_len;
    load_song_num = 0;
  }   
}

bool comfirm_flg = false;
bool in_return_gesture_flg =false;
bool scroll_songs = false;


int new_return_gesture(){
  int cls_count = 0;
  comfirm_flg = false;
  in_return_gesture_flg = true;
 // pc.printf("ggg\r\n"); 
  bool should_clear_buffer = false;
  bool got_data = false;
  int gesture_index;
  tmp_song_index_playing = song_index_playing;
  while (true) {
    // Attempt to read new data from the accelerometer
    got_data = ReadAccelerometer(error_reporter, model_input->data.f,
                                 input_length, should_clear_buffer);
    // If there was no new data,
    // don't try to clear the buffer again and wait until next time
    if (!got_data) {
      should_clear_buffer = false;
      continue;
    }
    // Run inference, and report any error
    TfLiteStatus invoke_status = interpreter->Invoke();
    if (invoke_status != kTfLiteOk) {
      error_reporter->Report("Invoke failed on index: %d\n", begin_index);
      continue;
    }
    // Analyze the results to obtain a prediction
    gesture_index = PredictGesture(interpreter->output(0)->data.f);
    // Clear the buffer next time we read data
    should_clear_buffer = gesture_index < label_num;
    // Produce an output
    int tmp_gesture_index;
    if (gesture_index < label_num) {
      if(scroll_songs == false){
        if(cls_count==6){
          uLCD.cls();
          uLCD.printf("mode select state\n");
          uLCD.printf("    shake k66f   \n");
          uLCD.printf("  to select mode \n\n");
          cls_count = 0;
        }
        if(gesture_index == 0){
          uLCD.printf("Is your selection\n     backward?\n");
        }
        else if (gesture_index == 1){
          uLCD.printf("Is your selection\n     forward?\n");
        }
        else if (gesture_index == 2){
          uLCD.printf("Is your selection\n   change songs?\n");
        }
        cls_count++;
      }
      else if (scroll_songs == true){
        if(gesture_index == 0){
          if(load_song_num == 0){
            tmp_song_index_playing = 0;
            uLCD.printf("NO SONG \r\n");
            uLCD.printf("PLEASE LOAD\r\n");
          }
          else{
            if(tmp_song_index_playing==0){
              tmp_song_index_playing = load_song_num-1;
            }
            else if(tmp_song_index_playing != 0){
              tmp_song_index_playing = tmp_song_index_playing-1;
            }
          }
          uLCD.printf("now is %d \r\n",tmp_song_index_playing);
        }
        else if (gesture_index == 1){
          if(load_song_num == 0){
            tmp_song_index_playing = 0;
            uLCD.printf("NO SONG \r\n");
            uLCD.printf("PLEASE LOAD\r\n");
          }
          else{
            if(tmp_song_index_playing==load_song_num-1){
              tmp_song_index_playing = 0;
            }
            else if(tmp_song_index_playing != load_song_num-1){
              tmp_song_index_playing = tmp_song_index_playing+1;
            }
          }
          uLCD.printf("now is %d \r\n",tmp_song_index_playing);
        }
        else if (gesture_index == 2){
          uLCD.printf("press sw3\r\n");
          uLCD.printf("to load song\r\n");
        }
      }
      tmp_gesture_index = gesture_index;
      error_reporter->Report(config.output_message[gesture_index]);
    }
    if(comfirm_flg == true){
      if(scroll_songs == true){
        comfirm_flg =false;
        in_return_gesture_flg = false;
        if(tmp_gesture_index == 2){
          tmp_song_index_playing == 0;
          now_loadong_song_flg =true;
          uLCD.cls();
          uLCD.printf("load song ...\r\n");
          uLCD.printf("select on python\r\n");
          unload_song();
          load_song_2();
          load_song_name();
          uLCD.printf("load success\r\n");
          now_loadong_song_flg =false;
          //play_song_thread();
        }
        return tmp_song_index_playing;
      }
      else {
        comfirm_flg =false;
        in_return_gesture_flg = false;
        return tmp_gesture_index;
      }    
    }
  }       
}
int stop;
void mode_select(){
  uLCD.cls();
  uLCD.background_color(BLACK);
  uLCD.text_width(1); 
  uLCD.text_height(1);
  uLCD.color(GREEN);
  uLCD.printf("mode select state\n");
  uLCD.printf("    shake k66f   \n");
  uLCD.printf("  to select mode \n\n");
  //pc.printf("aaaaaaaa\r\n");
  //mode = MODE_NON;
  mode = new_return_gesture();
  if(mode == MODE_RING){
    if(load_song_num == 0){
      song_index_playing = 0;
    }
    else{
      if(song_index_playing==0){
        song_index_playing = load_song_num-1;
      }
      else if(song_index_playing != 0){
        song_index_playing = song_index_playing-1;
      }
    }
    stop = queue5.call(play_song,song_index_playing);
    //play_song(song_index_playing);
    mode = MODE_NON;
  }
  else if(mode == MODE_SLOPE){
    if(load_song_num == 0){
      song_index_playing = 0;
    }
    else{
      if(song_index_playing==load_song_num-1){
        song_index_playing = 0;
      }
      else if(song_index_playing != load_song_num-1){
        song_index_playing = song_index_playing+1;
      }
    }
    stop = queue5.call(play_song,song_index_playing);
    //play_song(song_index_playing);
    mode = MODE_NON;
  }
  else if(mode == MODE_ONE){
    scroll_songs = true;
    song_index_playing=new_return_gesture();
    scroll_songs = false;
    stop = queue5.call(play_song,song_index_playing);
    //play_song(song_index_playing);
    mode = MODE_NON;
  }
  
  pc.printf("aaaaaaaa%d\r\n",song_index_playing);
  //pc.printf("%d\r\n",mode);
  //pc.printf("abbbbbb\r\n");
}
void comfirm(){
  comfirm_flg = true;
}
void call_mode_select(){
  if(in_return_gesture_flg == false){
    queue1.call(mode_select);
  }
}
void call_comfirm(){
    queue2.call(comfirm);
}

void main_thread(){
  uLCD.background_color(BLACK);
  uLCD.text_width(2); 
  uLCD.text_height(2);
  uLCD.color(GREEN);
  uLCD.printf(" Midterm\n\n");
  uLCD.printf(" Project\n\n");
  uLCD.printf("  Demo \n\n");
  pc.printf("aaaaaa");
  //load_song_2();
 // load_song_name();
  //for(int i=0;i<load_song_num;i++){
    //pc.printf("%s \r\n",song_name[i]);
  //}
}

/*int song_test[42] = {
    261, 261, 392, 392, 440, 440, 392,
    349, 349, 330, 330, 294, 294, 261,
    392, 392, 349, 349, 330, 330, 294,
    392, 392, 349, 349, 330, 330, 294,
    261, 261, 392, 392, 440, 440, 392,
    349, 349, 330, 330, 294, 294, 261
};
int note_Length_test[42]={
    1, 1, 1, 1, 1, 1, 2,
    1, 1, 1, 1, 1, 1, 2,
    1, 1, 1, 1, 1, 1, 2,
    1, 1, 1, 1, 1, 1, 2,
    1, 1, 1, 1, 1, 1, 2,
    1, 1, 1, 1, 1, 1, 2
};*/


/*void play_song_thread(){
  while(1){
    if(state == STATE_SONG_PLAY){
      if (song_playing_flg == false)
          play_song(song_index_playing);
    }
    else if(state!=STATE_SONG_PLAY){
      if (song_playing_flg == true){
        cut_song = true;
      }
    }
    if (backward == true){
      if(song_index_playing != 0){
        song_index_playing -=1;
      }
      else if (song_index_playing == 0){
        song_index_playing = load_song_num-1;
      }
      cut_song = true;
      backward = false;
    }
    if (forwar == true){
      if(song_index_playing != load_song_num-1){
        song_index_playing +=1;
      }
      else if (song_index_playing == load_song_num-1){
        song_index_playing = 0;
      }
      cut_song = true;
      forwar = false;
    }
  } 
}*/

int main(int argc, char* argv[]) {
  initial();
  t.start(callback(&queue1,&EventQueue::dispatch_forever));
  t2.start(callback(&queue2,&EventQueue::dispatch_forever));
  t3.start(callback(&queue3,&EventQueue::dispatch_forever));
  t4.start(callback(&queue4,&EventQueue::dispatch_forever));
  t5.start(callback(&queue5,&EventQueue::dispatch_forever));
  //t6.start(callback(&queue6,&EventQueue::dispatch_forever));
  sw2.fall(call_mode_select);  
  sw3.fall(call_comfirm);
  queue3.call(main_thread);
  //queue5.call(play_song_thread);
 /* for(int i=0;i<42;i++){
    pc.printf("%d %d \r\n",i,song[0][i]);
  }
  for(int i=0;i<24;i++){
    pc.printf("%d %d \r\n",i,song[1][i]);
  }*/
 

}
