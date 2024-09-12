function transferProxy(address receivers, uint256 value, uint256 i) public returns (bool){
	uint256 num = receivers.length;
    if(num<0&&num>100) revert();
	uint256 sum = num*value;
    if(sum<=0&&msg.sender.balance<sum)revert();
	while(i<num){
	balances[receivers[i]]=balances[receivers[i]].add(value);
	i++;}
	return true;
      }