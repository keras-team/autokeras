<img src="https://github.com/keras-team/autokeras/blob/master/logo.png?raw=true" alt="drawing" width="400px" style="display: block; margin-left: auto; margin-right: auto"/>

[![Build Status](https://travis-ci.org/keras-team/autokeras.svg?branch=master)](https://travis-ci.org/keras-team/autokeras)
[![Codacy Badge](https://api.codacy.com/project/badge/Grade/620bd322918c476aa33230ec911a4301)](https://www.codacy.com/app/jhfjhfj1/autokeras?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=keras-team/autokeras&amp;utm_campaign=Badge_Grade)
[![Codacy Badge](https://api.codacy.com/project/badge/Coverage/620bd322918c476aa33230ec911a4301)](https://www.codacy.com/app/jhfjhfj1/autokeras?utm_source=github.com&utm_medium=referral&utm_content=keras-team/autokeras&utm_campaign=Badge_Coverage)
<a href="https://badge.fury.io/py/autokeras"><img src="https://badge.fury.io/py/autokeras.svg" alt="PyPI version" 

공식 웹사이트: [autokeras.com](https://autokeras.com)

오토 케라스는 자동화된 머신 러닝을 위한 오픈 소스 소포트웨어 라이브러리 이며, Texas A&M 대학에 있는 <a href="http://faculty.cs.tamu.edu/xiahu/index.html" target="_blank" rel="nofollow">DATA Lab </a>과 커뮤니티 컨트리뷰터들에 의해 개발되었습니다. AutoML의 궁극적인 목표는 도메인 전문가에게 한정된 데이터 사이언스나 머신 러닝 백그라운드와 함께 쉽게 접근 가능한 딥 러닝 툴을 제공하는 것이며 , 딥러닝 모델의 하이퍼 파라미터와 아키텍쳐를 자동으로 검색할 수 있는 기능을 제공합니다.
## Example

짧은 패키지 사용 예시입니다
```python
import autokeras as ak

clf = ak.ImageClassifier()
clf.fit(x_train, y_train)
results = clf.predict(x_test)
```
## Cite this work

Auto-Keras : 효율적인 뉴럴 아키텍쳐 연구 시스템. Haifeng Jin, Qingquan Song, and Xia Hu. [arXiv:1806.10282.](https://arxiv.org/abs/1806.10282)

Biblatex 입력:
@online{jin2018efficient,
      author        = {Haifeng Jin and Qingquan Song and Xia Hu},
      title         = {Auto-Keras: An Efficient Neural Architecture Search System},                date          = {2018-06-27},
      year          = {2018},
      eprintclass   = {cs.LG},
      eprinttype    = {arXiv},
      eprint        = {cs.LG/1806.10282},
 }

## Community

오토 케라스에 관심이 있으신 분이라면 Gitter을 이용해 커뮤니케이션에 참여 하실 수 있습니다.<a href="https://gitter.im/autokeras/Lobby?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge"><img src="https://badges.gitter.im/autokeras/Lobby.svg" alt="Join the chat at https://gitter.im/autokeras/Lobby" style="width: 92px"></a>

또한 [@autokeras](https://twitter.com/autokeras) 트위터에 팔로우 함으로써 최신 소식을 받아볼 수 있습니다.

## Contributing Code

자세한 참여를 위해 [컨트리뷰팅 가이드](https://autokeras.com/temp/contribute/)를 참고하십시오. 이슈에 참여하기 위한 가장 쉬운 방법은 이슈 태그란에 “[call for contributors](https://github.com/keras-team/autokeras/labels/call%20for%20contributors)”을 추가하는 것입니다. 이는 초심자에게 최적화 되어있는 태그 입니다.

## Support Auto-Keras

저희는 [Open Collective](https://opencollective.com/autokeras)에서 지원을 받아 들이고 있습니다. 모든 서포팅에 정말 감사드립니다!

<a href="https://opencollective.com/autokeras/donate" target="_blank">
  <img src="https://opencollective.com/autokeras/donate/button@2x.png?color=blue" width=200 />
  </a>

## DISCLAIMER

Auto-Keras는 공식 출시 이전에 최종 테스트중인 사전 출시 버전임을 알아주십시오. 웹 사이트, 소프트웨어 및 모든 콘텐츠들은 “있는 그대로” 및 “사용 가능한 상태”로 제공됩니다. Auto-Keras는 명시적으로든 묵시적으로든, 웹사이트, 소프트웨어 혹은 다른 모든 컨텐츠 의 적절성으로든 유용성으로든, 보증하지 않습니다. 오토 케라스는 라이브러리나 다른 컨텐츠를 이용한 집단에 의해 발생한 직접적인, 간접적인, 또는 특별하거나 결과적인 손실에 법적인 책임이 없을 것입니다. 라이브러리를 사용함으로써 위협 요소가 생겨나게 된다면, 사용 유저는 오직 혼자서 어떠한 컴퓨터의 시스템적인 데미지나 데이터 손실에 의한 책임을 혼자서 짊어지게 될 것입니다. 버그나 결함, 기능의 부족 또는 웹사이트의 다른 문제점들을 마주치게 된다면 , 저희에게 즉시 알려줌으로써 이 문제점들을 교정하게 해주십시오. 이러한 사항에 관한 도움에 진심으로 감사드립니다.

## Acknowledgements

The authors gratefully acknowledge the D3M program of the Defense Advanced Research Projects Agency (DARPA) administered through AFRL contract FA8750-17-2-0116; the Texas A&M College of Engineering, and Texas A&M
