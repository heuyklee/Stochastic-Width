#!/bin/sh

# 실험을 위한 shell script를 작성한다.
# 수행할 작업의 순서는 다음과 같다.
# 1) ./SpatialConvolution2.lua 내부에 version = 1, bR = 0 
#     l26: local BYPASS_RATE = 0.2 (0~1, float)
#     l27: local BYPASS_VERSION = 2 (1 or 2)
# 2) th main.lua 실행
# 3) ./exData/ 아래 v$version_d$depth_br$br/$iter 폴더를 생성
# 4) 생성한 폴더 내부 결과 learningLog.t7 latest.t7 optimState_100.t7 model_100.t7
#    4개를 저장  
# 5) 2)~4)를 총 5회 반복

# 6) 1)로 돌아가 bR = 0.2로 수정
# 7) 2)~5) 반복

# 8) 6)~7)을 bR = 0.4/0.6/0.8 로 수정하며 반복(현재 상태로 봐서는 bR이 아주
#    작아야 제대로 수렴할 것으로 보여 정확한 수치는 수정 필요)

# 9) 1)로 돌아가 version = 2, bR = 0로 수정
# 10) 1)~8)을 반복

# 위의 모든 과정에서 각 과정 종료 시 log파일에 기록 남기도록 
