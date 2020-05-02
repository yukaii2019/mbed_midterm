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

/*#include "fsl_port.h"
#include "fsl_gpio.h"
#include "math.h"
#define UINT14_MAX        16383
// FXOS8700CQ I2C address
#define FXOS8700CQ_SLAVE_ADDR0 (0x1E<<1) // with pins SA0=0, SA1=0
#define FXOS8700CQ_SLAVE_ADDR1 (0x1D<<1) // with pins SA0=1, SA1=0
#define FXOS8700CQ_SLAVE_ADDR2 (0x1C<<1) // with pins SA0=0, SA1=1
#define FXOS8700CQ_SLAVE_ADDR3 (0x1F<<1) // with pins SA0=1, SA1=1
// FXOS8700CQ internal register addresses
#define FXOS8700Q_STATUS 0x00
#define FXOS8700Q_OUT_X_MSB 0x01
#define FXOS8700Q_OUT_Y_MSB 0x03
#define FXOS8700Q_OUT_Z_MSB 0x05
#define FXOS8700Q_M_OUT_X_MSB 0x33
#define FXOS8700Q_M_OUT_Y_MSB 0x35
#define FXOS8700Q_M_OUT_Z_MSB 0x37
#define FXOS8700Q_WHOAMI 0x0D
#define FXOS8700Q_XYZ_DATA_CFG 0x0E
#define FXOS8700Q_CTRL_REG1 0x2A
#define FXOS8700Q_M_CTRL_REG1 0x5B
#define FXOS8700Q_M_CTRL_REG2 0x5C
#define FXOS8700Q_WHOAMI_VAL 0xC7*/

#define MODE_RING 0
#define MODE_SLOPE 1
#define MODE_ONE 2
#define MODE_NON 3
DA7212 audio;
//I2C i2c( PTD9,PTD8);
Serial pc(USBTX, USBRX);
uLCD_4DGL uLCD(D1, D0, D2);
//int m_addr = FXOS8700CQ_SLAVE_ADDR1;
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
Thread t5(osPriorityHigh);
Thread t6(osPriorityNormal);
Thread t7(osPriorityNormal);


EventQueue queue1(32*EVENTS_EVENT_SIZE);
EventQueue queue2(32*EVENTS_EVENT_SIZE);
EventQueue queue3(32*EVENTS_EVENT_SIZE);
EventQueue queue4(32*EVENTS_EVENT_SIZE);
EventQueue queue5(32*EVENTS_EVENT_SIZE);
EventQueue queue6(32*EVENTS_EVENT_SIZE);
EventQueue queue7(32*EVENTS_EVENT_SIZE);

InterruptIn sw2(SW2);
InterruptIn sw3(SW3);
DigitalOut greenled(LED2);
Timer debounce_sw2;
Timer debounce_sw3;
Timer timer;
Timer timer_2;
int** song;
int** note_len;
int** taiko;
char** song_name;
char serialInBuffer[32];
int serialCount;
int load_song_num = 0;
int* song_len;
int* song_speed;
int* correct;

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
bool load_song_flg = false;
bool taiko_flg = false;
int n;
int freq = 0;
int now_taiko_note = 0;
int note_i = 0;
int taiko_or_not = 0;
int tmp_taiko_or_not = 0;
int max_song_len = -100;
bool queue7finish = false;
bool comfirm_flg = false;
bool in_return_gesture_flg =false;
bool scroll_songs = false;

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
void taiko_display2(float last_time){
  uLCD.cls();
  timer_2.reset();
  while(timer_2.read()<last_time){
    if(now_taiko_note == 1){
        uLCD.filled_rectangle(0,0,10,10,RED);
    }
    else if(now_taiko_note == 2){
        uLCD.filled_rectangle(0,0,10,10,BLUE);
    }
    else{
        uLCD.filled_rectangle(0,0,10,10,BLACK);
    }
    if(TAIKO_NOTE == 1){
        if(timer.read_ms()>200){
            uLCD.filled_circle(64,64,50,RED);
            uLCD.filled_circle(64,64,30,BLACK);
            uLCD.filled_circle(64,64,50,BLACK);
            timer.reset();
        }
    }

    else if (TAIKO_NOTE == 2){
        if(timer.read_ms()>200){
            uLCD.filled_circle(64,64,50,BLUE);
            uLCD.filled_circle(64,64,30,BLACK);
            uLCD.filled_circle(64,64,50,BLACK);
            timer.reset();
        }
    }
  }
  timer_2.reset();  
  uLCD.filled_rectangle(0,0,10,10,BLACK);
  queue7finish = true;
} 
void playNote(){
    for(int i=0;i<kAudioTxBufferSize;i++){
        waveform[i] = (int16_t) (sin((double)i * 2. * M_PI / (double) (kAudioSampleFrequency / freq)) * ((1<<16) - 1));
    }
    for(int j=0;j<kAudioSampleFrequency / kAudioTxBufferSize;++j){
        audio.spk.play(waveform,kAudioTxBufferSize);
    }
    if(taiko_or_not == 1){
      if(now_taiko_note ==1){
        if(TAIKO_NOTE == 2 && correct[note_i] == 0){
          correct[note_i] = 2;
        }
        else if(TAIKO_NOTE==1&& correct[note_i] == 0){
          correct[note_i] = 1;
        }
      }
      else if(now_taiko_note ==2){
        if(TAIKO_NOTE == 2 && correct[note_i] == 0){
          correct[note_i] = 1;
        }
        else if(TAIKO_NOTE==1&& correct[note_i] == 0){
          correct[note_i] = 2;
        }
      }   
      else if(now_taiko_note == 0 && correct[note_i] == 0){
          correct[note_i] = 3;
      }    
      // if(TAIKO_NOTE == 1){
      //   uLCD.cls();
      //   uLCD.background_color(RED);
      //   uLCD.cls();
      //   //wait(0.1);
      //   uLCD.background_color(BLACK);
      //   uLCD.cls();
      // }
      // if(TAIKO_NOTE == 2){
      //   uLCD.cls();
      //   uLCD.background_color(BLUE);
      //   uLCD.cls();
      //   //wait(0.1);
      //   uLCD.background_color(BLACK);
      //   uLCD.cls();
      // }
    }
}
void play_song(int index){
  float k = (song_len[index])*1./16;
  float speed = song_speed[index]*1./100;
  float wait_time;
  float a = k;
  float last_time = 0;    
  for(int i=0;i<song_len[index];i++){
      last_time += note_len[index][i];
  }
  last_time = last_time*speed;
  if (in_return_gesture_flg == false && now_loadong_song_flg == false){
    uLCD.cls();
    uLCD.textbackground_color(BLACK);
    uLCD.printf("Song Name:\r\n\r\n%s\r\n\r\n",song_name[song_index_playing]);    
    uLCD.printf("Song Length:%1.0f s\r\n\r\n",last_time);
    if(taiko_or_not == 1){
      uLCD.text_width(4);
      uLCD.text_height(4);
      uLCD.color(RED);
      for (int i=5; i>=0; --i) {
          uLCD.locate(1,2);
          uLCD.printf("%2D",i);
          wait(1);
      }
    }
  }
  uLCD.text_width(1);
  uLCD.text_height(1);
  uLCD.color(GREEN);
  song_playing_flg = true;
  idC=queue4.call_every(1,playNote);
  if(taiko_or_not == 1){
    queue7.call(taiko_display2,last_time);
  }      
  for(int i=0;i<song_len[index];i++){  
    pc.printf("%d\r\n",taiko[index][i]);   
    freq=song[index][i];
    now_taiko_note = taiko[index][i];
    wait_time = note_len[index][i]*speed;
    note_i = i;
    wait(wait_time-0.05);
    freq = 0;
    wait(0.05);
    if(i+1>=a){
      if (in_return_gesture_flg == false && now_loadong_song_flg == false && taiko_or_not == 0){
        uLCD.printf("*");
      }
      a+=k;
    } 
    if(cut_song == true){
      break;
    }
    
  }
  if(song_index_playing == load_song_num-1 && cut_song != true && taiko_or_not == 0){
    song_index_playing = 0;
  }
  else if (song_index_playing<load_song_num-1 && cut_song != true && taiko_or_not == 0){
    song_index_playing ++;
  }
  note_i = 0;
  if(taiko_or_not == 0 && cut_song != true){
    queue5.call(play_song,song_index_playing);
  }
  if(taiko_or_not == 1){
    while(1){
      if(queue7finish == true){
        queue7finish =false;
        break;
      }
      else{
        wait(0.5);
      }
    }
    int c=0;
    int b=0;
    for (int i=0;i<song_len[index];i++){
        pc.printf("%d ",correct[i]);
        uLCD.printf("%d ",correct[i]);
        if(correct[i] == 1){
          c++;
        }
        else if (correct[i] == 2 or correct[i] == 0){
          b++;
        } 
        correct[i] = 0;
    }
    uLCD.printf("\r\nCorrect: %d\r\n\r\n",c);
    uLCD.printf("Bad: %d\r\n\r\n",b);
    uLCD.printf("Point: %f\r\n\r\n",(c*1./(c+b))*100);
  }
  taiko_or_not = 0; 
  song_playing_flg = false;
  cut_song = false;
} 
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
  i = 0;
  while(i<len){
    if(pc.readable()){
      serialInBuffer[serialCount] = pc.getc();
      serialCount++;
      if(serialCount == 4){
        serialInBuffer [serialCount] = '\0';
        taiko[index][i] = (int)atof(serialInBuffer);
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
          if(song_len[i]>max_song_len){
            max_song_len = song_len[i];
          }
          serialCount = 0;
          break;
        }
      }
    }
  }
  song_speed = new int[load_song_num];
  for (int i=0;i<load_song_num;i++){
    while(1){
      if(pc.readable()){
        serialInBuffer[serialCount] = pc.getc();
        serialCount++;
        if(serialCount == 4){
          serialInBuffer [serialCount] = '\0';
          song_speed[i] = (int)atof(serialInBuffer);
          serialCount = 0;
          break;
        }
      }
    }
  }
  correct = new int [max_song_len];
  for(int i=0;i<max_song_len;i++){
    correct [i] = 0;
  }
  song = new int* [load_song_num];
  for(int i=0;i<load_song_num;i++){
    song[i] = new int[song_len[i]];
  }
  note_len = new int* [load_song_num];
  for(int i=0;i<load_song_num;i++){
    note_len[i] = new int[song_len[i]];
  }
  taiko = new int* [load_song_num];
  for(int i=0;i<load_song_num;i++){
    taiko[i] = new int[song_len[i]];
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

    delete [] song_speed;
    delete [] song_len;
    delete [] correct;
    load_song_num = 0;
    mode = MODE_NON;
    song_playing_flg = false;
    song_index_playing = 0;
    tmp_song_index_playing = 0;
    cut_song = false;
    taiko_flg = false;
    freq = 0;
    now_taiko_note = 0;
    note_i = 0;
    taiko_or_not = 0;
    tmp_taiko_or_not = 0;
    max_song_len = -100;
    queue7finish = false;
  }   
}
void print_song_list_page(){
  int page = 0;
  uLCD.cls();
  uLCD.textbackground_color(BLACK);
  uLCD.printf("   Scroll Song    \r\n");
  if(load_song_num == 0){
    uLCD.printf("NO SONG \r\n");
    uLCD.printf("PLEASE LOAD\r\n");
  }
  if((tmp_song_index_playing/3) == 0){
    page = 0;
  }
  else if((tmp_song_index_playing/3) == 1){
    page = 1;
  }
  else if((tmp_song_index_playing/3) == 2){
    page = 2;
  }
  else if((tmp_song_index_playing/3) == 3){
    page = 3;
  }

  if(page == 0){
    for(int i=0;i<3;i++){
      if(i==load_song_num){
        break;
      }
      if(i==tmp_song_index_playing){
        uLCD.textbackground_color(BLUE);
      }
      uLCD.printf("%d.\r\n%s\r\n\r\n",i+1,song_name[i]);
      uLCD.textbackground_color(BLACK);
    }
  }
  if(page == 1){
    for(int i=3;i<6;i++){
      if(i==load_song_num){
        break;
      }
      if(i==tmp_song_index_playing){
        uLCD.textbackground_color(BLUE);
      }
      uLCD.printf("%d.\r\n%s\r\n\r\n",i+1,song_name[i]);
      uLCD.textbackground_color(BLACK);
    }
  }
  if(page == 2){
    for(int i=6;i<9;i++){
      if(i==load_song_num){
        break;
      }
      if(i==tmp_song_index_playing){
        uLCD.textbackground_color(BLUE);
      }
      uLCD.printf("%d.\r\n%s\r\n\r\n",i+1,song_name[i]);
      uLCD.textbackground_color(BLACK);
    }
  }
  if(page == 3){
    for(int i=9;i<12;i++){
      if(i==load_song_num){
        break;
      }
      if(i==tmp_song_index_playing){
        uLCD.textbackground_color(BLUE);
      }
      uLCD.printf("%d.\r\n%s\r\n\r\n",i+1,song_name[i]);
      uLCD.textbackground_color(BLACK);
    }
  }
}
void print_taiko_or_not(){
  uLCD.cls();
  uLCD.textbackground_color(BLACK);
  uLCD.printf("Song Name:\r\n\r\n%s\r\n\r\n",song_name[song_index_playing]);
  uLCD.printf("Song Length:%d\r\n\r\n",song_len[song_index_playing]);
  uLCD.printf("Shack K66F To\r\nSelect Taiko Or\r\nSong Playing mode\r\n\r\n");
  if(tmp_taiko_or_not == 0){
    uLCD.textbackground_color(BLUE);
    uLCD.printf("Song Playing \r\n");
    uLCD.textbackground_color(BLACK);
    uLCD.printf("Taiko \r\n");
  }
  else if (tmp_taiko_or_not == 1){
    uLCD.printf("Song Playing \r\n");
    uLCD.textbackground_color(BLUE);
    uLCD.printf("Taiko \r\n");
    uLCD.textbackground_color(BLACK);
  }
}
void print_forward_backward_change_songs(int type){
  uLCD.cls();
  uLCD.textbackground_color(BLACK);
  uLCD.printf("mode select state\n");
  uLCD.printf("    shake k66f   \n");
  uLCD.printf("  to select mode \n\n");
  if(type == 0){
    if(load_song_num == 0){
      uLCD.printf("NO SONG \r\n");
      uLCD.printf("PLEASE LOAD\r\n");
    }
    else{
      uLCD.printf("Now Playing:\r\n\r\n%d\r\n%s\r\n\r\n",song_index_playing+1,song_name[song_index_playing]);
      uLCD.printf("Backward To:\r\n\r\n");
      if(song_index_playing==0){
        uLCD.printf("%d.\r\n%s\r\n\r\n",load_song_num,song_name[load_song_num-1]);
      }
      else if(song_index_playing != 0){
        uLCD.printf("%d.\r\n%s\r\n\r\n",song_index_playing,song_name[song_index_playing-1]);
      }
    }
  }
  else if (type == 1){
    if(load_song_num == 0){
      uLCD.printf("NO SONG \r\n");
      uLCD.printf("PLEASE LOAD\r\n");
    }
    else{
      uLCD.printf("Now Playing:\r\n\r\n%d.\r\n%s\r\n\r\n",song_index_playing+1,song_name[song_index_playing]);
      uLCD.printf("Forward To:\r\n\r\n");
      if(song_index_playing==load_song_num-1){
        uLCD.printf("%d.\r\n%s\r\n\r\n",1,song_name[0]);
      }
      else if(song_index_playing != load_song_num-1){
        uLCD.printf("%d.\r\n%s\r\n\r\n",song_index_playing+2,song_name[song_index_playing+1]);
      }
    }
  }
  else if (type == 2){
    if(load_song_num == 0){
      uLCD.printf("NO SONG \r\n");
      uLCD.printf("PLEASE LOAD\r\n\r\n");
      uLCD.printf("Confirm To Load\r\nSongs\r\n");
    }
    else {
      uLCD.printf("Now Playing:\r\n\r\n%d.\r\n%s\r\n\r\n",song_index_playing+1,song_name[song_index_playing]);
      uLCD.printf("change song\r\n");
    } 
  }
}
int new_return_gesture(){
  int cls_count = 0;
  int scroll_page = 0;
  comfirm_flg = false;
  in_return_gesture_flg = true;
 // pc.printf("ggg\r\n"); 
  bool should_clear_buffer = false;
  bool got_data = false;
  int gesture_index;
  tmp_taiko_or_not = 0;
  tmp_song_index_playing = song_index_playing;
  if(scroll_songs == true){
    print_song_list_page();
  }
  else if (taiko_flg == true){
    print_taiko_or_not();
  }
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
      if (scroll_songs == true){  
        if(gesture_index == 0){
          if(load_song_num == 0){
            tmp_song_index_playing = 0;
          }
          else{
            if(tmp_song_index_playing==0){
              tmp_song_index_playing = load_song_num-1;
            }
            else if(tmp_song_index_playing != 0){
              tmp_song_index_playing = tmp_song_index_playing-1;  
            }
            if((tmp_song_index_playing/3) == 0){
              scroll_page = 0;
            }
            print_song_list_page();
          }        
        }
        else if (gesture_index == 1){
          if(load_song_num == 0){
            tmp_song_index_playing = 0;
          }
          else{
            if(tmp_song_index_playing==load_song_num-1){
              tmp_song_index_playing = 0;
            }
            else if(tmp_song_index_playing != load_song_num-1){
              tmp_song_index_playing = tmp_song_index_playing+1;
            }
            print_song_list_page();
          }       
        }
        else if (gesture_index == 2){
          uLCD.cls();
          uLCD.printf("press sw3\r\n");
          uLCD.printf("to load song\r\n");
        }
      }
      else if (taiko_flg == true){
        if(gesture_index == 2){
          if(tmp_taiko_or_not == 0){
            tmp_taiko_or_not = 1;
            print_taiko_or_not();
          }
          else if (tmp_taiko_or_not == 1){
            tmp_taiko_or_not = 0;
            print_taiko_or_not();
          }
        }
      }
      else {
        // uLCD.cls();
        // uLCD.printf("mode select state\n");
        // uLCD.printf("    shake k66f   \n");
        // uLCD.printf("  to select mode \n\n");
        if(gesture_index == 0){
          //uLCD.printf("Is your selection\n     backward?\n");
          print_forward_backward_change_songs(0);
        }
        else if (gesture_index == 1){
          //uLCD.printf("Is your selection\n     forward?\n");
          print_forward_backward_change_songs(1);
        }
        else if (gesture_index == 2){
          //uLCD.printf("Is your selection\n   change songs?\n");
          print_forward_backward_change_songs(2);
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
          if(song_playing_flg==true){
            cut_song =true;
          }
          uLCD.cls();
          uLCD.printf("load song ...\r\n");
          uLCD.printf("select on python\r\n");
          unload_song();
          load_song_2();
          load_song_name();
          uLCD.printf("load success\r\n");
          now_loadong_song_flg =false;
          load_song_flg = true;
          //play_song_thread();
        }
        return tmp_song_index_playing;
      }
      else if (taiko_flg == true){
        comfirm_flg =false;
        in_return_gesture_flg = false;
        return tmp_taiko_or_not;
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
      taiko_flg = true;
      if(song_playing_flg==true){
          cut_song =true;
      }
      taiko_or_not = new_return_gesture();
      if(taiko_or_not == 1){
        comfirm_flg = false;
        uLCD.cls();
        uLCD.printf("press SW3 to start game\r\n");
        while(1){
          if(comfirm_flg == true){
            break;
          }
          else{
            wait(0.5);
          }
        }
        comfirm_flg = false;
        stop = queue5.call(play_song,song_index_playing);
      }
      else if (taiko_or_not ==0){
        stop = queue5.call(play_song,song_index_playing);
      }
      taiko_flg = false;
    }
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
      taiko_flg = true;
      if(song_playing_flg==true){
          cut_song =true;
      }
      taiko_or_not = new_return_gesture();
      if(taiko_or_not == 1){
        comfirm_flg = false;
        uLCD.cls();
        uLCD.printf("press SW3 to start game\r\n");
        while(1){
          if(comfirm_flg == true){
            break;
          }
          else{
            wait(0.5);
          }
        }
        comfirm_flg = false;
        stop = queue5.call(play_song,song_index_playing);
      }
      else if (taiko_or_not ==0){
        stop = queue5.call(play_song,song_index_playing);
      }
      taiko_flg = false;
    }  
    mode = MODE_NON;
  }
  else if(mode == MODE_ONE){
    scroll_songs = true;
    song_index_playing=new_return_gesture();
    scroll_songs = false;
    if (load_song_flg == true){
      load_song_flg = false;
    }
    else {    
      taiko_flg = true;
      if(song_playing_flg==true){
          cut_song =true;
      }
      taiko_or_not = new_return_gesture();
      if(taiko_or_not == 1){
        comfirm_flg = false;
        uLCD.cls();
        uLCD.printf("press SW3 to start game\r\n");
        while(1){
          if(comfirm_flg == true){
            break;
          }
          else{
            wait(0.5);
          }
        }
        comfirm_flg = false;
        stop = queue5.call(play_song,song_index_playing);
      }
      else if (taiko_or_not ==0){
        stop = queue5.call(play_song,song_index_playing);
      }
      taiko_flg = false;      
    } 
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
  if(debounce_sw2.read_ms()>1000){
    if(in_return_gesture_flg == false){
      queue1.call(mode_select);
    }
  }
  debounce_sw2.reset();
}
void call_comfirm(){
  if(debounce_sw3.read_ms()>1000){
    queue2.call(comfirm);
  }
  debounce_sw3.reset();
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


int main(int argc, char* argv[]) {
  initial();
  greenled = 1;
  timer.start();
  timer_2.start();
  debounce_sw2.start();
  debounce_sw3.start();
  t.start(callback(&queue1,&EventQueue::dispatch_forever));
  t2.start(callback(&queue2,&EventQueue::dispatch_forever));
  t3.start(callback(&queue3,&EventQueue::dispatch_forever));
  t4.start(callback(&queue4,&EventQueue::dispatch_forever));
  t5.start(callback(&queue5,&EventQueue::dispatch_forever));
  t6.start(callback(&queue6,&EventQueue::dispatch_forever));
  t7.start(callback(&queue7,&EventQueue::dispatch_forever));
  sw2.fall(call_mode_select);  
  sw3.fall(call_comfirm);
  queue3.call(main_thread);
  queue6.call(accelerometer);
  //queue5.call(play_song_thread);
 /* for(int i=0;i<42;i++){
    pc.printf("%d %d \r\n",i,song[0][i]);
  }
  for(int i=0;i<24;i++){
    pc.printf("%d %d \r\n",i,song[1][i]);
  }*/
 

}
