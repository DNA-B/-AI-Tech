import React, { useState, useRef, useEffect, useContext } from 'react';
import { ScrollView } from 'react-native';
import styled, { useTheme } from 'styled-components/native';
import CalendarPicker from 'react-native-calendar-picker';
import axios from 'axios';
import Container from '../components/Container';
import Header from '../components/Header';
import Icon from '../components/Icon';
import { Context } from '../utils/Context';
import { SCREEN_WIDTH, w8, w14, w16, w28, w32, w48, w64, w84, w96, w108, w144 } from '../utils/theme';
import { toast } from '../utils/toast';

const getLastDates = yesterday => {
	const date = new Date(yesterday);
	date.setMonth(date.getMonth() - 1);
	date.setDate(date.getDate() + 1);

	const lastDates = [];
	for (; date <= yesterday; date.setDate(date.getDate() + 1)) {
		lastDates.push(new Date(date));
	}

	return lastDates;
};

const getMonthDates = selectedDate => {
	const date = new Date(selectedDate);
	date.setDate(1);

	const monthDates = [];
	for (; date.getMonth() === selectedDate.getMonth(); date.setDate(date.getDate() + 1)) {
		monthDates.push(new Date(date));
	}

	return monthDates;
};

const dayToKorean = date => {
	const korean = ['일', '월', '화', '수', '목', '금', '토'];

	return korean[date.getDay()];
};

const CalendarModal = ({ visible, onSelectCalendar, onClose, yesterday, selectedDate }) => {
	const theme = useTheme();

	return (
		<CalendarModalContainer visible={visible} transparent animationType="fade">
			<CalendarModalBackground activeOpacity={1} onPress={onClose}>
				<CalendarContainer>
					<CalendarPicker
						onDateChange={onSelectCalendar}
						weekdays={['월', '화', '수', '목', '금', '토', '일']}
						months={['1월', '2월', '3월', '4월', '5월', '6월', '7월', '8월', '9월', '10월', '11월', '12월']}
						maxDate={yesterday}
						width={SCREEN_WIDTH * 0.9}
						textStyle={{ fontFamily: 'Light' }}
						previousTitle='◀'
						nextTitle='▶'
						selectMonthTitle='월 선택: '
						selectYearTitle='연 선택'
						initialDate={selectedDate}
						selectedStartDate={selectedDate}
						selectedDayColor={theme.secondary}
						selectedDayTextColor={theme.background}
						todayBackgroundColor={theme.background}
						todayTextStyle={{ color: theme.gray }}
						disabledDatesTextStyle={{ color: theme.gray }}
					/>
				</CalendarContainer>
			</CalendarModalBackground>
		</CalendarModalContainer>
	);
};

const Feed = () => {
	const [yesterday, setYesterday] = useState(() => {
		const yesterday = new Date();
		yesterday.setDate(yesterday.getDate() - 1);
		return yesterday;
	});
	const [selectedDate, setSelectedDate] = useState(yesterday);
	const [dates, setDates] = useState(getLastDates(yesterday));
	const [feeds, setFeeds] = useState([]);
	const [calendarVisible, setCalendarVisible] = useState(false);
	const [datesScrollOffset, setDatesScrollOffset] = useState(0);

	const { userId } = useContext(Context);

	const datesScrollRef = useRef(null);
	const theme = useTheme();

	useEffect(() => {
		datesScrollToRight();
		// selectedDate에 해당하는 feeds 받아오기
		loadFeeds(yesterday);
	}, []);

	useEffect(() => {
		const selectedIndex = dates.findIndex(date => date.getDate() === selectedDate.getDate());
		const selectedX = selectedIndex * (w108 + w28) + w64;

		const startIndex = datesScrollOffset + w64;
		const endIndex = datesScrollOffset + w64 + 6 * (w108 + w28);

		if (selectedX < startIndex) {
			datesScrollRef.current.scrollTo({ x: selectedX - w64, y: 0, animated: true });
		}
		else if (selectedX > endIndex) {
			datesScrollRef.current.scrollTo({ x: selectedX - 6 * (w108 + w28) - w64, y: 0, animated: true });
		}
	}, [selectedDate]);

	const datesScrollToRight = () => {
		if (datesScrollRef.current) {
			datesScrollRef.current.scrollToEnd();
		}
	};

	const loadFeeds = async (date) => {
		const dateStr = `${date.getFullYear()}-${String(date.getMonth() + 1).padStart(2, '0')}-${String(date.getDate()).padStart(2, '0')}`;

		try {
			const res = await axios.post('http://34.64.120.166:8000/api/sum-message/', {
				user_id: userId,
				date: dateStr
			}, {
				headers: {
					'Content-Type': 'application/x-www-form-urlencoded'
				}
			})
			
			const resFeeds = res.data.map(feed => ({
				time: `${String(new Date(feed.start_time).getHours()).padStart(2, '0')}:${String(new Date(feed.start_time).getMinutes()).padStart(2, '0')}`,
				imageUrls: feed.image_link,
				text: feed.stylechangemessage
			}))
			setFeeds(resFeeds);

		} catch (error) {
			toast('일기를 불러오지 못했습니다.');
			console.log(error);
		}
	};

	const onSelectDate = date => {
		setSelectedDate(date);
		loadFeeds(date);
	};

	const onSelectCalendar = selectedCalendarDate => {
		let date = new Date(selectedCalendarDate);
		
		const startDate = new Date(dates[0].getFullYear(), dates[0].getMonth(), dates[0].getDate());
		const endDate = new Date(dates[dates.length - 1].getFullYear(), dates[dates.length - 1].getMonth(), dates[dates.length - 1].getDate());
		date = new Date(date.getFullYear(), date.getMonth(), date.getDate());
		
		if (date.getMonth() === yesterday.getMonth()) {
			setDates(getLastDates(yesterday));
		}
		else if (date < startDate || date > endDate) {
			setDates(getMonthDates(date));
		}
		
		loadFeeds(date);
		setCalendarVisible(false);
		setSelectedDate(date);
	};

	const onGotoYesterday = () => {
		setDates(getLastDates(yesterday));
		setSelectedDate(yesterday);
		loadFeeds(yesterday);
	};

	return (
		<Container>
			<Header view='feed' />

			<DatesScrollContainer>
				<DatesScroll
					horizontal
					showsHorizontalScrollIndicator={false}
					ref={datesScrollRef}
					onScroll={e => setDatesScrollOffset(e.nativeEvent.contentOffset.x)}
				>
					{dates.map((date, idx) => (
						<DateContainer
							key={idx}
							date={date}
							selectedDate={selectedDate}
							onPress={() => onSelectDate(date, idx)}
						>
							<DayText date={date} selectedDate={selectedDate}>
								{dayToKorean(date)}
							</DayText>
							<DateText date={date} selectedDate={selectedDate}>
								{date.getDate()}
							</DateText>
						</DateContainer>
					))}
				</DatesScroll>
			</DatesScrollContainer>

			<SelectedDateContainer>
				<SelectedDate onPress={() => setCalendarVisible(true)}>
					<SelectedDateText>
						{selectedDate.getFullYear()}. {selectedDate.getMonth() + 1}. {selectedDate.getDate()}. ▾
					</SelectedDateText>
				</SelectedDate>
				<YesterdayButton onPress={onGotoYesterday}>
					<Icon source={require('../assets/today-icon.png')} />
				</YesterdayButton>
			</SelectedDateContainer>

			<CalendarModal
				visible={calendarVisible}
				onSelectCalendar={onSelectCalendar}
				onClose={() => setCalendarVisible(false)}
				yesterday={yesterday}
				selectedDate={selectedDate}
			/>

			<FeedsScroll>
				{feeds.length !== 0 ? (feeds.map((feed, idx) => (
					<FeedContainer key={idx}>
						<FeedTime>{feed.time}</FeedTime>
						{feed.imageUrls.length !== 0 && (
							<Carousel>
								{feed.imageUrls.map((uri, idx) => (
									<CarouselImageContainer key={idx}>
										<CarouselPaging>{idx + 1}/{feed.imageUrls.length}</CarouselPaging>
										<CarouselImage source={{ uri }} resizeMode="cover" />
									</CarouselImageContainer>
								))}
							</Carousel>
						)}
						<FeedText
							selectable
							selectionColor={theme.secondaryBackground}>
							{feed.text}
						</FeedText>
					</FeedContainer>
				))) : (
					<NoticeText>이 날짜에 저장된 일기가 없습니다.</NoticeText>
				)}
			</FeedsScroll>
		</Container>
	);
};

const DatesScrollContainer = styled.View`
	height: ${w96 * 2}px;
	border-bottom-width: 1px;
	border-bottom-color: ${({ theme }) => theme.gray};
`;

const DatesScroll = styled(ScrollView).attrs({
	contentContainerStyle: {
		paddingRight: w64,
		paddingLeft: w64
	}
})`
	flex: 1;
	flex-direction: row;
	padding-top: ${w16}px;
`;

const DateContainer = styled.TouchableOpacity`
	flex: 1;
	justify-content: space-between;
	align-items: center;
	width: ${w108}px;
	height: ${w144}px;
	margin: ${w16}px ${w14}px;
	padding: ${w16}px 0;
	border-radius: ${w48}px;

	background-color: ${({ date, selectedDate, theme }) => 
		date.getDate() === selectedDate.getDate() ? theme.secondary : theme.background
	};
`;

const DayText = styled.Text`
	font-size: 11px;
	font-family: Light;
	color: ${({ date, selectedDate, theme }) => 
		date.getDate() === selectedDate.getDate() ? theme.background : 'black'
	};
`;

const DateText = styled.Text`
	font-size: 21px;
	font-family: Regular;
	color: ${({ date, selectedDate, theme }) => 
		date.getDate() === selectedDate.getDate() ? theme.background : 'black'
	};
`;

const SelectedDateContainer = styled.View`
	flex-direction: row;
	justify-content: space-between;
	align-items: center;
	padding: ${w16}px ${w64}px ${w28}px ${w28}px;
	border-bottom-width: 1px;
	border-bottom-color: ${({ theme }) => theme.gray};
`;

const SelectedDate = styled.TouchableOpacity``;

const SelectedDateText = styled.Text`
	font-size: ${w96}px;
	font-family: Regular;
	line-height: ${w108 + w16}px;
`;

const YesterdayButton = styled.TouchableOpacity`
	width: ${w96}px;
	height: ${w96}px;
	background-color: ${({ theme }) => theme.secondaryBackground};
	border-radius: ${w32}px;
	margin-top: ${w16}px;
	padding: ${w8}px;
`;

const CalendarModalContainer = styled.Modal``;

const CalendarModalBackground = styled.TouchableOpacity`
	flex: 1;
	justify-content: center;
	background-color: rgba(0, 0, 0, 0.5);
`;

const CalendarContainer = styled.View`
	margin: 16px;
	background-color: ${({ theme }) => theme.background};
	border-radius: 16px;
	padding: 16px;
	height: 42%;
`;

const FeedsScroll = styled.ScrollView`
	flex: 1;
	padding: ${w32}px 0;
`;

const FeedContainer = styled.View`
	flex: 1;
	padding: ${w32}px 0;
`;

const Carousel = styled(ScrollView).attrs(() => ({
	horizontal: true,
	pagingEnabled: true,
	showsHorizontalScrollIndicator: false
}))`
	margin-top: ${w28}px;
`;

const CarouselImageContainer = styled.View`
	align-items: center;
	width: ${SCREEN_WIDTH}px;
	height: ${SCREEN_WIDTH}px;
`;

const CarouselImage = styled.Image`
	width: ${SCREEN_WIDTH}px;
	height: ${SCREEN_WIDTH}px;
`;

const CarouselPaging = styled.Text`
	padding: ${w28}px ${w32}px;
	color: ${({ theme }) => theme.background};
	background-color: ${({ theme }) => theme.secondaryBackground}80;
	font-family: Light;
	border-radius: 100px;
	position: absolute;
	z-index: 1;
	top: ${w32}px;
	right: ${w32}px;
`;

const FeedTime = styled.Text`
	padding-left: ${w48}px;
	padding-bottom: ${w8}px;
	font-size: ${w84}px;
	font-family: Light;
`;

const FeedText = styled.Text`
	margin: ${w28}px 0 ${w64}px;
	padding: 0 ${w96}px;
	font-size: ${w64}px;
	font-family: HandWriting;
`;

const NoticeText = styled.Text`
	margin: ${w96}px;
	text-align: center;
	font-size: 17px;
	font-family: Regular;
`;

export default Feed;