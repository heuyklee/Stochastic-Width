-- 함수의 선언 순서에 따른 영향을 확인하기 위한 테스트 파일
-- > 굳이 사용 순서에 맞게 선언할 필요는 없다는 결론

function foo1(a)
   return a + 1
end

function foo2(b)
   return foo3(b)
end

function foo3(c)
   return c+10
end


print(foo2(1))
