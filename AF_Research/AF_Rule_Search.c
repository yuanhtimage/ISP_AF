#define AF_THRESHOLD_Args0 0.8
#define AF_THRESHOLD_Args1 0.004
#define INITIAL_STEP 1
#define COARSE_STEP 8
#define MID_STEP 3
#define FINE_STEP 1

typedef enum {
  INITIAL = INITIAL_STEP,
  COARSE = COARSE_STEP,
  MID = MID_STEP,
  FINE = FINE_STEP
} FOCUS_CONTROL_AREA_TypeDef;

typedef struct{
  unsigned int fv_cur;
  unsigned int pos_cur;
} WinItem;

typedef enum {
  FOCUS_INIT,
  FOCUS_READY,
  FOCUS_ING
} FOCUS_STATUS_TypeDef;


void autoFocus(unsigned int pos_begin, unsigned int pos_end){
  // 0. Init something
  unsigned long preTime = GetTimeStampMs();

  unsigned int pos_cur = pos_begin;
  unsigned int pos_best = pos_begin;
  AFMoveToPosition(pos_begin);
  int count_it = 0;  // Iteration counter
  int count_down = 0;  // Down counter
  FOCUS_CONTROL_AREA_TypeDef AREA = INITIAL;
  WinItem moving_win[5];
  unsigned int moving_win_idx = 0;
  unsigned int moving_win_sum = 0;
  unsigned int moving_win_avg = 0;
  unsigned int moving_win_pre = 0;
  unsigned int moving_win_max = 0;
  unsigned int fv_cur = 0;
  KSAFGetFv(&fv_cur, 1);
  while (pos_begin <= pos_cur && pos_cur < pos_end){
    // 1. Divide the area
    if (count_it <= 5){
      printf("INIT: %d %d %d\n", pos_cur, fv_cur, AREA);
      AREA = INITIAL;
    }else{
      //printf("RUNNING: %d %d %d\n", pos_cur, moving_win_avg, AREA);
      if (moving_win_avg <= AF_THRESHOLD_Args0 * moving_win_max){
        AREA = COARSE;
        count_down = 0;
      }else{
        int fv_diff = moving_win_avg - moving_win_pre;
        if (fv_diff > AF_THRESHOLD_Args1 * moving_win_pre){
          AREA = FINE;
          count_down = 0;
        }else if (AREA == FINE && fv_diff > 0){
          count_down = 0;
        }else if (fv_diff < 0){
          if (AREA == FINE){
            count_down++;
          }
          if (count_down == 3){
            AREA = MID;
            count_down = 0;
          }
        }else{
          AREA = MID;
          count_down = 0;
        }
      }
    }

    // 2. Moving and sliding windows
    count_it++;
    pos_cur += AREA;
    AFMoveToPosition(pos_cur);
    KSAFGetFv(&fv_cur, 0);

    if (moving_win_idx == 5){
      moving_win_sum -= moving_win[0].fv_cur;
      for (int i = 0; i < 5 - 1; i++){
        moving_win[i] = moving_win[i + 1];
      }
      moving_win[moving_win_idx - 1].fv_cur = fv_cur;
      moving_win[moving_win_idx - 1].pos_cur = pos_cur;
      moving_win_sum += fv_cur;
      moving_win_pre = moving_win_avg;
      moving_win_avg = moving_win_sum / 5;
      if (moving_win_avg > moving_win_max){
        moving_win_max = moving_win_avg;
        pos_best = moving_win[5 / 2].pos_cur;
      }
    }else{
      moving_win[moving_win_idx].fv_cur = fv_cur;
      moving_win[moving_win_idx++].pos_cur = pos_cur;
      moving_win_sum += fv_cur;
    }
  }
  
}