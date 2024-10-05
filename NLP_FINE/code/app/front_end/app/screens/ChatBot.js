import React, { useState, useRef, useEffect, useContext } from 'react';
import { TextInput } from 'react-native';
import styled, { useTheme } from 'styled-components/native';
import { LinearGradient } from 'expo-linear-gradient';
import * as ImagePicker from 'expo-image-picker';
import * as SecureStore from 'expo-secure-store';
import axios from 'axios';
import Container from '../components/Container';
import Header from '../components/Header';
import Icon from '../components/Icon';
import { Context } from '../utils/Context';
import { w8, w16, w24, w28, w32, w48, w64, w96 } from '../utils/theme';
import { toast } from '../utils/toast';

const FIRSTCHAT = '안녕하세요! 무슨 일이 있으셨나요?';

const ChatBot = () => {
	const [messages, setMessages] = useState([]);
	const [inputText, setInputText] = useState('');
	const [imageUrls, setImageUrls] = useState([]);
	const scrollViewRef = useRef(null);
	const { userId, isChatting, setIsChatting } = useContext(Context);
	const theme = useTheme();

	useEffect(() => {
		loadChatLog();
		scrollToBottom();

		(async () => {
			const { status } = await ImagePicker.requestMediaLibraryPermissionsAsync();
			if (status !== 'granted') {
				toast('사진 접근 권한이 필요합니다.');
			}
		})();
	}, []);

	useEffect(() => {
		scrollToBottom();
	}, [messages]);

	const scrollToBottom = () => {
		if (scrollViewRef.current) {
			scrollViewRef.current.scrollToEnd();
		}
	};

	const loadChatLog = async () => {
		try {
			const res = await axios.get(`http://34.64.120.166:8000/api/chat-messages/?user_id=${userId}`,
				{},
				{
					headers: {
						'Content-Type': 'application/x-www-form-urlencoded'
					}
				});

			setMessages(res.data.map((message, idx) => ({
				id: idx,
				message: message.message,
				sender: message.sender,
				created_at: new Date(message.created_at)
			})));

		} catch (error) {
			toast('서버 접속이 원활하지 않습니다.');
			console.log(error);
		}

	};

	const onEventStart = async () => {
		try {
			const res = await axios.post('http://34.64.120.166:8000/api/chat-messages/', {  
				user_id: userId,
				message: null,
				start: 1
			}, {  
				headers: {
					'Content-Type': 'application/x-www-form-urlencoded'
				}
			});

			const res_message = {
				id: Date.now(),
				message: res.data.message,
				sender: 'bot',
				created_at: new Date(res.data.created_at)
			};

			setMessages((prevMessages) => [...prevMessages, res_message]);
			setImageUrls([]);
			setIsChatting(true);
			await SecureStore.setItemAsync('isChatting', 'true');

		} catch (error) {
			toast('서버 접속이 원활하지 않습니다.');
			console.log(error);
		}
	};

	const onEventEnd = async () => {
		try {
			await axios.post('http://34.64.120.166:8000/api/chat-messages/', {  
				user_id: userId,
				message: null,
				end: true
			}, {
				headers: {
					'Content-Type': 'application/x-www-form-urlencoded'
				}
			});

			scrollToBottom();
			setInputText('');
			setIsChatting(false);
			await SecureStore.setItemAsync('isChatting', 'false');

		} catch (error) {
			toast('서버 접속이 원활하지 않습니다.');
			console.log(error);
		}
	};

	const handleSend = async () => {
		if (!inputText)
			return;
		try {
			const message = {
				id: Date.now(),
				message: inputText,
				sender: 'user',
				created_at: new Date()
			};
			setMessages((prevMessages) => [...prevMessages, message]);
			setInputText('');
			
			const res = await axios.post('http://34.64.120.166:8000/api/chat-messages/', {  
				user_id: userId,
				message: inputText,
				start: null
			}, {
				headers: {
					'Content-Type': 'application/x-www-form-urlencoded'
				}
			});

			const res_message = {
				id: Date.now(),
				message: res.data.message,
				sender: 'bot',
				created_at: res.data.created_at
			};
			
			setMessages((prevMessages) => [...prevMessages, res_message]);

		} catch (error) {
			toast('서버 접속이 원활하지 않습니다.');
			console.log(error);
		}
	};

	const handleImage = async () => {
		try {
			let result = await ImagePicker.launchImageLibraryAsync({
				mediaTypes: ImagePicker.MediaTypeOptions.Images,
				allowsMultipleSelection: true,
				base64: true,
				quality: 1,
			});

			if (!result.canceled) {
				if (result.assets.length > 5) {
					toast('사진은 5장까지 업로드 가능합니다.');
					return;
				}

				const images = result.assets.map(r => r.base64);

				const res = await axios.post('http://34.64.120.166:8000/api/image/', {
					images: JSON.stringify(images),
					user_id: userId
				}, {
					headers: {
						'Content-Type': 'application/x-www-form-urlencoded'
					}
				})
				
				setImageUrls(res.data.urls);
			}

		} catch (error) {
			toast('서버 접속이 원활하지 않습니다.');
			console.log(error);
		}
	};

	return (
		<Container>
			<Header view='chatbot' />

			<MessagesContainer ref={scrollViewRef}>
				{messages.map((message, idx) => (
					<MessageContainer key={message.id}>
						{(message.sender === 'bot' && message.message === FIRSTCHAT) && (
							<EventSeparator>
								<EventSeparatorText>
									{message.created_at.getFullYear()}년{' '}
									{message.created_at.getMonth() + 1}월{' '}
									{message.created_at.getDate()}일{' '}
									{message.created_at.getHours()}시{' '}
									{message.created_at.getMinutes()}분
								</EventSeparatorText>
							</EventSeparator>
						)}
						<MessageContent key={message.id} sender={message.sender}>
							{message.sender === 'bot' && (
								<BotProfile>
									<Logo source={require('../assets/logo-small.png')} />
								</BotProfile>
							)}
							<Message key={message.id} sender={message.sender}>
								<MessageText
									selectable
									selectionColor={theme.secondaryBackground}>
										{message.message}
								</MessageText>
							</Message>
							{(isChatting && message.sender === 'bot' && idx === messages.length - 1) && (
								<EndButtonContainer>
									<EndButton onPress={onEventEnd}>
										<EndButtonText>이 주제의{'\n'}대화 끝내기</EndButtonText>
									</EndButton>
								</EndButtonContainer>
							)}
						</MessageContent>
					</MessageContainer>
				))}
			</MessagesContainer>

			<BottomContainer>
				{!isChatting ? (
					<NoticeContainer>
						{messages.length !== 0 ? (
							<>
								{imageUrls.length === 0 ? (
									<>
										<NoticeText>
											대화가 종료되었습니다.{'\n'}
											이 시간을 추억할 만한 사진이 있나요?
										</NoticeText>
										<WideButton onPress={handleImage}>
											<WideButtonText>기록에 사진 추가하기</WideButtonText>
										</WideButton>
									</>
								) : (
									<>
										<ImagesContainer>
											{imageUrls.map(url => (
												<UploadedImage source={{ uri: url }} key={url} />
											))}
										</ImagesContainer>
										<NoticeText>
											일기에 사진이 추가되었습니다.{'\n'}
											새로운 대화를 시작해보세요!
										</NoticeText>
									</>
								)}
							</>
						) : (
							<NoticeText>
								안녕하세요, Fine 챗봇입니다.{'\n'}
								대화를 시작해 수다를 떨어보세요! {'\n'}
								오늘 밤, Fine가 멋진 일기를 써드립니다.
							</NoticeText>
						)}
						<WideButton onPress={onEventStart}>
							<WideButtonText>새로운 대화 시작하기</WideButtonText>
						</WideButton>
					</NoticeContainer>
				) : (
					<InputContainer>
						<Input
							value={inputText}
							multiline={true}
							onChangeText={setInputText}
							onFocus={scrollToBottom}
						/>
						<SendButton onPress={handleSend}>
							<Icon source={require('../assets/send-icon.png')} />
						</SendButton>
					</InputContainer>
				)}
			</BottomContainer>
		</Container>
	);
};

const MessagesContainer = styled.ScrollView`
	flex: 1;
	padding: ${w28}px ${w28}px ${w64}px;
`;

const MessageContainer = styled.View`
	flex: 1;
`;

const EventSeparator = styled.View`
	flex: 1;
	flex-direction: row;
	justify-content: center;
	margin: ${w28}px 0;
`;

const EventSeparatorText = styled.Text`
	padding: ${w8}px ${w64}px;
	border-radius: 100px;
	background-color: ${({ theme }) => theme.primaryBackground};
	color: ${({ theme }) => theme.background};
	font-family: Light;
	font-size: ${w32}px;
`;

const MessageContent = styled.View`
	flex: 1;
	flex-direction: row;
	justify-content: ${props => (props.sender === 'user' ? 'flex-end' : 'flex-start')};
	margin-bottom: ${w28}px;
`;

const BotProfile = styled.View`
	width: ${w96}px;
	height: ${w96}px;
	margin-right: ${w28}px;
	border-radius: 14px;
	background-color: ${({ theme }) => theme.secondary};
	justify-content: center;
	align-items: center;
`;

const Logo = styled.Image`
	width: ${w64}px;
	height: ${w64}px;
`;

const Message = styled.View`
	background-color: ${({ theme, sender }) => (sender === 'user' ? theme.primary : theme.secondary)};
	border-radius: ${w48}px;
	padding: ${w24}px ${w28}px;
	max-width: ${({ sender }) => (sender === 'user' ? 66.7 : 58.5)}%;
	min-width: ${w96}px;
`;

const MessageText = styled.Text`
	color: white;
	font-size: 16px;
	font-family: Light;
`;

const BottomContainer = styled.View`
	align-items: center;
	background-color: ${({ theme }) => theme.background};
`;

const NoticeContainer = styled.View`
	align-items: center;
	padding-bottom: ${w64}px;
	width: 100%;
`;

const NoticeText = styled.Text`
	margin: ${w96}px;
	text-align: center;
	font-size: 17px;
	font-family: Regular;
`;

const ImagesContainer = styled.View`
	flex-direction: row;
	justify-content: center;
	background-color: ${({ theme }) => theme.secondaryBackground}80;
	margin-top: ${w32}px;
	padding: ${w16}px;
`;

const UploadedImage = styled.Image`
	width: ${w96 * 2}px;
	height: ${w96 * 2}px;
	margin: ${w16}px;
	border-radius: ${w16}px;
`;

const WideButton = styled.TouchableOpacity`
	align-items: center;
	background-color: ${({ theme }) => theme.secondaryBackground}F0;
	border-radius: 100px;
	margin-bottom: ${w32}px;
	padding: ${w32}px;
	width: 70%;
`;

const WideButtonText = styled.Text`
	font-size: 16px;
	color: ${({ theme }) => theme.background};
	font-family: Light;
`;

const EndButtonContainer = styled.View`
	justify-content: center;
	margin-left: ${w32}px;
`

const EndButton = styled.TouchableOpacity`
	background-color: ${({ theme }) => theme.secondaryBackground}D0;
	border-radius: 100px;
	padding: ${w16}px ${w32}px;
`;

const EndButtonText = styled.Text`
	text-align: center;
	font-size: 10px;
	color: ${({ theme }) => theme.background};
	font-family: Light;
`;

const InputContainer = styled(LinearGradient).attrs(({ theme }) => ({
	colors: [theme.secondaryBackground, theme.primaryBackground],
	start: { x: 0, y: 0 },
	end: { x: 1, y: 1 },
}))`
	flex-direction: row;
	align-items: center;
	margin: ${w28}px;
	padding: ${w16}px;
	padding-right: ${w28}px;
	border-radius: 24px;
`;

const Input = styled(TextInput).attrs(({ theme }) => ({
	cursorColor: theme.secondaryBackground,
	selectionColor: theme.secondaryBackground
}))`
	flex: 1;
	background-color: ${({ theme }) => theme.background};
	border-radius: 20px;
	padding: ${w16}px ${w48}px;
	font-size: 16px;
	font-family: Regular;
`;

const SendButton = styled.TouchableOpacity`
	width: ${w96}px;
	height: ${w96}px;
	margin-left: ${w16}px;
	padding: ${w16}px;
`;

export default ChatBot;