이 버전은 SpatialConvolution2.lua 파일 내부에서 gradInput을 normalize하는 부분까지의
구현을 마친 버전이다.

하지만 실행 결과 약 23-24% 정도의 오차를 보였으며, 이는 gradInput이 정확히 전달되는 것이
아니라 중간에 normalize를 거치며 정보의 손실이 존재하기 때문이라고 판단하였다.

따라서 다음 버전에서는 BK를 하나만 1로 set 시키는 방식이 아니라,
해당 BK를 이전 layer의 weight를 복사해서 가져오고, output역시 복사해서 가져오는 방식으로
구현을 진행할 예정이다. 그리고 backward시 gradWeight를 앞의 gradWeight에 더해주는 방식으로...
============== 20161119_1400 backup

weight, output 을 뒷 layer에 그대로 복사해서 가지고 있고,
backward의 결과로 생성된 gradWeight를 앞 layer에 더해주거나, 복사해주거나
하는 형식으로의 구현... 결과 제대로 나오지 않음

다음 버전은 gradInput을 넘겨줘 보는 것으로...
그런데 ReLU와 BN의 경우에 gradInput을 어떻게 처리 하는지 확인 필요...?
============== 20161120_1400 backup

대부분의 구현이 제대로 동작하지 않아서 현재는 weight만 뒷 layer에 복사해서
학습 시키고 gradWeight를 앞으로 더해주거나, 복사하거나 등의 실험을 해보았고
성능이 12%정도에 머무르는 것을 확인하였다.
============== 20161121_1800 backup

기존 아이디어 구현을 위해 SpatialConvolution2.lua 내 updateOutput 함수 안에서
forward 시 이전 conv layer의 output 값을 그대로 복사해 오도록 수정하였고,
determineBypass를 epoch 단위로 실행, backward 전에 BKzero() 실행 등의 시도를 하였음
9~11% 정도의 에러를 보이는 것을 확인, 구현에는 문제가 없는 것으로 생각 되고,
빠르게 실행해볼 수 있는 다른 아이디어들을 구현하여 실험해보는 것이 좋을 것 같음
============== 20161122_1600 backup

models/resnet.lua 내에서 setBypassRate() 를 통해서 2/2,4/2,4,6 만 0.5로 set하고
나머지는 SpatialConvolution2.lua 내에서 모두 bR을 0으로 set 하여 진행해 보았으며
결과는 original resnet과 유사하거나 약간 떨어지는 정도인 8.5~9% 에러를 보였다.
이제 gradWeight를 뒤에서 가져와서 더해주는 방식이 아닌 Select.lua와 같이
gradOutput을 넘겨주는 방식으로 구현을 해보려 한다.
구체적으로는 bR이 0이 아닌 conv의 BK위치에 해당하는 gradOutput을 이전 layer의
gradOutput에 더해주거나, 복사하거나, 평균을 구하는 식으로 구현을 진행해볼 계획
============== 20161122_1900 backup
