import React, { createContext, useState } from 'react';

export const Context = createContext();

export const ContextProvider = ({ children }) => {
	const [isLogin, setIsLogin] = useState(false);
	const [userId, setUserId] = useState('');
	const [isChatting, setIsChatting] = useState(false);

	return (
		<Context.Provider value={{ isLogin, setIsLogin, userId, setUserId, isChatting, setIsChatting }}>
			{children}
		</Context.Provider>
	);
};