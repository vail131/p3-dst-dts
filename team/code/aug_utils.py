import random
import copy
def kortime_var(time:str):
    ori_time=time
    time=list(map(int,time.split(':')))
    hour=time[0]
    minute=time[1]
    if hour<10:
        hours='0'+str(hour)
    if minute<10 and minute!=0:
        minutes='0'+str(minute)
    
    minutes=minute
    minute='반' if minute==30 else str(minute)+'분'
    if minute=='0분':
        minute=''
        
    time_arr=set([f'{hour}시 {minute}', f'{hour}시{minute}',ori_time,f'{hour}시{minute}',f'{hour}시 {minute}'])
    if hour<9:
        time_arr.add(f'새벽 {hour}시 {minute}')
        time_arr.add(f'새벽{hour}시 {minute}')
        time_arr.add(f'새벽{hour}시{minute}')
    elif hour<12:
        time_arr.add(f'오전 {hour}시 {minute}')
        time_arr.add(f'오전{hour}시 {minute}')
        time_arr.add(f'오전{hour}시{minute}')
    if hour>=12 and hour<18:
        time_arr.add(f'낮 {hour}시 {minute}')
        time_arr.add(f'낮 {hour}시{minute}')
        time_arr.add(f'오후{hour}시 {minute}')
        time_arr.add(f'오후{hour}시{minute}')
        time_arr.add(f'낮 {hour-12}시 {minute}')
        time_arr.add(f'낮 {hour-12}시{minute}')
        time_arr.add(f'오후{hour-12}시 {minute}')
        time_arr.add(f'오후{hour-12}시{minute}')
    elif hour>=18:
        time_arr.add(f'오후 {hour}시 {minute}')
        time_arr.add(f'오후{hour}시 {minute}')
        time_arr.add(f'오후{hour}시{minute}')
        time_arr.add(f'저녁 {hour}시 {minute}')
        time_arr.add(f'저녁{hour}시 {minute}')
        time_arr.add(f'저녁{hour}시{minute}')
        time_arr.add(f'오후 {hour-12}시 {minute}')
        time_arr.add(f'오후{hour-12}시 {minute}')
        time_arr.add(f'오후{hour-12}시{minute}')
        time_arr.add(f'저녁 {hour-12}시 {minute}')
        time_arr.add(f'저녁{hour-12}시 {minute}')
        time_arr.add(f'저녁{hour-12}시{minute}')
        time_arr.add(f'밤 {hour-12}시 {minute}')
        time_arr.add(f'밤{hour-12}시 {minute}')
        time_arr.add(f'밤{hour-12}시{minute}')
    if minutes==30:
        time_arr.add(f'{hour}시 {minutes}분')
        time_arr.add(f'{hour}시{minutes}분')
        if hour<9:
            time_arr.add(f'새벽 {hour}시 {minutes}분')
            time_arr.add(f'새벽{hour}시 {minute}분')
            time_arr.add(f'새벽{hour}시{minute}분')
        elif hour<12:
            time_arr.add(f'오전 {hour}시 {minutes}분')
            time_arr.add(f'오전{hour}시 {minutes}분')
            time_arr.add(f'오전{hour}시{minutes}분')
        if hour>=12 and hour<18:
            time_arr.add(f'낮 {hour}시 {minutes}분')
            time_arr.add(f'낮 {hour}시{minutes}분')
            time_arr.add(f'오후{hour}시 {minutes}분')
            time_arr.add(f'오후{hour}시{minutes}분')
        elif hour>=18:
            time_arr.add(f'오후 {hour}시 {minutes}분')
            time_arr.add(f'오후{hour}시 {minutes}분')
            time_arr.add(f'오후{hour}시{minutes}분')
            time_arr.add(f'저녁 {hour}시 {minutes}분')
            time_arr.add(f'저녁{hour}시 {minutes}분')
            time_arr.add(f'저녁{hour}시{minutes}분')
            time_arr.add(f'오후 {hour-12}시 {minutes}분')
            time_arr.add(f'오후{hour-12}시 {minutes}분')
            time_arr.add(f'오후{hour-12}시{minutes}분')
            time_arr.add(f'저녁 {hour-12}시 {minutes}분')
            time_arr.add(f'저녁{hour-12}시 {minutes}분')
            time_arr.add(f'저녁{hour-12}시{minutes}분')
            time_arr.add(f'밤 {hour-12}시 {minutes}분')
            time_arr.add(f'밤{hour-12}시 {minutes}분')
            time_arr.add(f'밤{hour-12}시{minutes}분')
    
    
    return sorted([x.strip() for x in list(time_arr)],key=lambda x:len(x),reverse=True)


def time2kor(time:str,option=None)-> str:
    """["12:30" 형식의 문자열 시간을 "12시 30분"으로 변경한다. 정각일 경우 "12시"만 리턴]

    Args:
        time (str): ["12:30" 형태의 시간]

    Returns:
        [str]: [문자열 시간 리턴]
    """
    ran=random.randrange(0,8)
    
    l=list(map(int,time.split(':')))
    #5시 00분 정각일 때 "5시" 이렇게 리턴함
    
    new_time=f'{l[0]}시'+(('') if l[1]==0 else f' {l[1]}분')
    if l[1]==30:
        new_time=new_time.replace(f' {l[1]}분',' 반') 
        if option :
            new_time=''.join(new_time.split())
        
    return new_time

def time2ampm(time:str,option=None,option2=None)-> str:
    """["12:30" 형식의 문자열 시간을 "오후/오전 12시 30분"으로 변경한다. 정각일 경우 "오전 12시"만 리턴]

    Args:
        time (str): ["12:30" 형태의 시간]

    Returns:
        str: [문자열 시간 리턴]
    """
    ran=random.randrange(0,8)
    
    l=list(map(int,time.split(':')))
    #5시 00분 정각일 때 "5시" 이렇게 리턴함
    hour=l[0]
    minute=l[1]
    
    new_time=(f'오전 {hour}시' if hour<12 else f"오후 {hour-12}시")+(('') if l[1]==0 else f' {l[1]}분')
    if l[1]==30:
        new_time=new_time.replace(f' {l[1]}분',' 반') 
        if option :
            new_time=''.join(new_time.split())
    if '오후' in new_time and option2==None:
        new_time=new_time.replace('오후','저녁')
        
    return new_time

def kor2time(kor:str)-> str:
    """["12시 30분"형식의 한글 문자열 시간 정보를 state에 적합한 "12:30"으로 변경한다]

    Args:
        kor (str): ["12시 30분"형식의 한글 문자열 시간 정보]

    Returns:
        str: ["12:30"]
    """
    
    kor=kor.split(' ')
    minute=''
    hour=kor[0][0:-1]
    if len(kor)>1:
        minute=str(int(kor[1][0:-1]))
    if int(hour)<10 :
        hour='0'+hour
    if len(minute)>0 and int(minute)<10 :
        minute='0'+minute
    return f'{hour}:{minute}'
        
def make_randtime(time:str)-> str:
    """[랜덤으로 "12:30" 형식의 시간을 생성한다]

    Args:
        time (str): ["12:30" 형태의 시간]

    Returns:
        str: [랜덤으로 생성한 시간]
    """
    
    hour=random.randrange(0,24)
    minute=random.randrange(0,60)
    hour=str(hour)
    minute=str(minute)
    
    if int(hour)<10 :
        hour='0'+hour
    if int(minute)<10 :
        minute='0'+minute
    return f'{hour}:{minute}'
    
def change_dialogue(dialogue:list,time_dict:dict,transfer_type:str)-> list:
    """[train_dials.json을 기반으로 시간을 추출하여 원본 dial을 2가지 방향("kor"/"ampm")으로 변형한다]

    Args:
        dialogue (list): [train_dials 중 특정 원소 원본]
        time_dict (dict(set)): [전달받은 dialogue에 포함된 모든 시간, set으로 중복이 없다]
        transfer_type (str): ['kor'/'ampm'을 선택해서 변환]

    Returns:
        list: [변형된 dialogue로 [{dialogue_idx:val},{domains:val},{dialogue:val}] 원본과 같은 형태를 지닌다]
    """
    #dialogue와 dialogue idx 변경해서 새로운 data를 리턴함
    new_data=copy.deepcopy(dialogue)
    new_dialogue=dialogue['dialogue']
    #oo시 oo분으로 변경
    if transfer_type=='kor':
        new_data['dialogue_idx']=f"{dialogue['dialogue_idx']}_kor"
        for dom_slot,time_set in time_dict.items():
            for time in time_set:
                new_time=make_randtime(time)
#                 print(time,new_time,time2kor(time))
                #모든 dom_slot의 state 시간을 새로운 시간으로 변경
                for ds in time_dict.keys():
                    new_dialogue=str(new_dialogue).replace(f'{ds}-{time}',f'{ds}-{new_time}')
                #아직 변경되지 않은 text에 존재하는 시간을 한글버전으로 변경
                for t in kortime_var(time):
                    if t in new_dialogue:
                        nt_var=kortime_var(new_time)
                        ran=random.randrange(0,len(nt_var))
                        new_dialogue=new_dialogue.replace(t,nt_var[ran])
                        break

    new_data['dialogue']=eval(new_dialogue)
    
    return new_data

def dict_checker(time_dict:dict)->bool:
    """[dict(set)의 내용물이 아무것도 없는 경우를 확인한다. 슬롯의 시간value가 dontcare일 때를 위함이다]

    Args:
        time_dict (dict(set)): [전달받은 dialogue에 포함된 모든 시간, set으로 중복이 없다]

    Returns:
        bool: [정상적으로 시간값들이 포함돼있어 변형가능한 경우 1, dontcare같은 예외가있어 전혀 바뀌지 않을 경우 0 반환]
    """
    count=sum(list(map(len,time_dict.values())))
    return 0 if count==0 else 1 

def augmentation(train_dials:list)->list:
    """[list로 불러온 train_dials.json를 받아 새로운 new_train_dials를 생성한다.]

    Args:
        train_dials (list): [원본 데이터 전체]

    Returns:
        list: [새로운 데이터]
    """
    new_dataset=[]
    cnt=0
    for dialogue in train_dials:
        if '식당' in dialogue['domains'] or '택시' in dialogue['domains'] or '지하철' in dialogue['domains']:
            time_dict=dict({'식당-예약 시간':set(), '택시-출발 시간':set(), '택시-도착 시간':set(), '지하철-출발 시간':set()})
            dial_dial=dialogue['dialogue']
            #도메인에서 식당,지하철,택시 없으면 거름
            for idx, dic in enumerate(dial_dial):
                if idx%2==0:
                    for sv in dic['state']:
                        if '식당-예약 시간' in sv or '택시-출발 시간' in sv or '택시-도착 시간' in sv or '지하철-출발 시간' in sv:
                            dom,slot,val=sv.split('-')
                            if val =='dontcare':
                                continue
                            for t in kortime_var(val):
                                if t in dic['text'] or t in dial_dial[idx-1]['text']:
                                    time_dict[f'{dom}-{slot}'].add(val)
                                    break
            
            if dict_checker(time_dict):
#                 ran=random.randrange(0,10)
#                 if ran<3:
                new_dataset.append(change_dialogue(dialogue,time_dict,'kor'))
#                 else :
#                     new_dataset.append(change_dialogue(dialogue,time_dict,'ampm'))
                cnt+=1
    #원본까지 합치고 싶을 때
    # new_dataset.extend(train_dials)
    print(f'-----------{cnt}개를 augmentation하였습니다----------')

    return new_dataset
