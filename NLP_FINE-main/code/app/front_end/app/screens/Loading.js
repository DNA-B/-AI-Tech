import { useEffect, useContext } from 'react';
import * as Application from 'expo-application';
import * as SecureStore from 'expo-secure-store';
import * as SplashScreen from 'expo-splash-screen';
import * as Font from 'expo-font';
import axios from 'axios';
import { Context } from '../utils/Context';
import { toast } from '../utils/toast';

const Loading = () => {
	const { setIsLogin, setUserId, setIsChatting } = useContext(Context);

	useEffect(() => {
		(async () => {
			try {
				// load font
				await Font.loadAsync({
					'Light': require('../assets/fonts/NanumSquareNeo-aLt.ttf'),
					'Regular': require('../assets/fonts/NanumSquareNeo-bRg.ttf'),
					'Bold': require('../assets/fonts/NanumSquareNeo-cBd.ttf'),
					'HandWriting': require('../assets/fonts/nanum_handwriting.ttf')
				});
				
				const savedUserId = await SecureStore.getItemAsync('id');
				
				if (savedUserId) {
					// login
					setUserId(savedUserId);
					
				} else {
					// signup
					const newUserId = Application.androidId;
					await SecureStore.setItemAsync('id', newUserId);
					
					await axios.post('http://34.64.120.166:8000/api/profile/', {
						user_id: newUserId
					}, {
						headers: {
							'Content-Type': 'application/x-www-form-urlencoded'
						}
					});

					setUserId(newUserId);
				}

				const isChatting = await SecureStore.getItemAsync('isChatting');
				setIsChatting(isChatting === 'true');
				setIsLogin(true);

				// hide splash screen
				await SplashScreen.hideAsync();

			} catch (error) {
				toast('로그인에 실패했습니다.');
				console.log(error);
			}
		})();

	}, []);

	return null;
};

export default Loading;