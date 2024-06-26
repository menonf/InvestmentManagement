/****** Object:  Schema [Analytics]    Script Date: 24/03/2024 22:23:58 ******/
CREATE SCHEMA [Analytics]
GO
/****** Object:  Table [Analytics].[PortfolioAnalytics]    Script Date: 24/03/2024 22:23:58 ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [Analytics].[PortfolioAnalytics](
	[PA_ID] [int] IDENTITY(1,1) NOT NULL,
	[PortfolioShortName] [varchar](50) NULL,
	[MetricName] [varchar](50) NULL,
	[MetricType] [varchar](50) NULL,
	[MetricLevel] [varchar](50) NULL,
	[MetricValue] [float] NULL,
 CONSTRAINT [PK_Portfolio] PRIMARY KEY CLUSTERED 
(
	[PA_ID] ASC
)WITH (STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
) ON [PRIMARY]
GO
/****** Object:  Table [dbo].[MarketData]    Script Date: 24/03/2024 22:23:58 ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[MarketData](
	[MD_ID] [int] IDENTITY(1,1) NOT NULL,
	[AsOfDate] [date] NOT NULL,
	[SecID] [int] NOT NULL,
	[Open] [float] NULL,
	[High] [float] NULL,
	[Low] [float] NULL,
	[Close] [float] NULL,
	[Volume] [float] NULL,
	[Dividends] [float] NULL,
	[Stock_Splits] [float] NULL,
	[Dataload_Date] [datetime] NULL,
 CONSTRAINT [PK_MarketData] PRIMARY KEY CLUSTERED 
(
	[MD_ID] ASC
)WITH (STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
) ON [PRIMARY]
GO
/****** Object:  Table [dbo].[Portfolio]    Script Date: 24/03/2024 22:23:58 ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[Portfolio](
	[PortID] [int] IDENTITY(1,1) NOT NULL,
	[PortfolioShortName] [varchar](50) NULL,
	[PortfolioName] [varchar](200) NULL,
	[PortfolioType] [varchar](200) NULL,
 CONSTRAINT [PK_Portfolio] PRIMARY KEY CLUSTERED 
(
	[PortID] ASC
)WITH (STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
) ON [PRIMARY]
GO
/****** Object:  Table [dbo].[PortfolioHoldings]    Script Date: 24/03/2024 22:23:58 ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[PortfolioHoldings](
	[HID] [int] IDENTITY(1,1) NOT NULL,
	[AsOfDate] [date] NULL,
	[PortID] [int] NULL,
	[SecID] [int] NULL,
	[HeldShares] [decimal](9, 4) NULL,
 CONSTRAINT [PK_PortfolioHoldings] PRIMARY KEY CLUSTERED 
(
	[HID] ASC
)WITH (STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
) ON [PRIMARY]
GO
/****** Object:  Table [dbo].[SecurityMaster]    Script Date: 24/03/2024 22:23:58 ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[SecurityMaster](
	[SecID] [int] IDENTITY(1,1) NOT NULL,
	[Ticker] [varchar](10) NULL,
	[Sector] [varchar](50) NULL,
	[Country] [varchar](50) NULL,
	[SecurityType] [nvarchar](50) NOT NULL,
	[SecurityName] [varchar](50) NULL,
 CONSTRAINT [PK_Securities_Master] PRIMARY KEY CLUSTERED 
(
	[SecID] ASC
)WITH (STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
) ON [PRIMARY]
GO

SET IDENTITY_INSERT [dbo].[Portfolio] ON 

INSERT [dbo].[Portfolio] ([PortID], [PortfolioShortName], [PortfolioName], [PortfolioType]) VALUES (1, N'GTEF', N'GlobalTech', N'Portfolio')
INSERT [dbo].[Portfolio] ([PortID], [PortfolioShortName], [PortfolioName], [PortfolioType]) VALUES (2, N'QQQ', N'Nasdaq100 ETF', N'ETF')
INSERT [dbo].[Portfolio] ([PortID], [PortfolioShortName], [PortfolioName], [PortfolioType]) VALUES (3, N'Nasdaq100', N'Nasdaq100', N'Benchmark')
INSERT [dbo].[Portfolio] ([PortID], [PortfolioShortName], [PortfolioName], [PortfolioType]) VALUES (4, N'MSCI', N'MSCI VaR', N'Portfolio')
SET IDENTITY_INSERT [dbo].[Portfolio] OFF
GO
SET IDENTITY_INSERT [dbo].[PortfolioHoldings] ON 

INSERT [dbo].[PortfolioHoldings] ([HID], [AsOfDate], [PortID], [SecID], [HeldShares]) VALUES (1, CAST(N'2023-10-02' AS Date), 1, 1, CAST(900.0000 AS Decimal(9, 4)))
INSERT [dbo].[PortfolioHoldings] ([HID], [AsOfDate], [PortID], [SecID], [HeldShares]) VALUES (2, CAST(N'2023-10-02' AS Date), 1, 2, CAST(800.0000 AS Decimal(9, 4)))
INSERT [dbo].[PortfolioHoldings] ([HID], [AsOfDate], [PortID], [SecID], [HeldShares]) VALUES (3, CAST(N'2023-10-02' AS Date), 1, 3, CAST(700.0000 AS Decimal(9, 4)))
INSERT [dbo].[PortfolioHoldings] ([HID], [AsOfDate], [PortID], [SecID], [HeldShares]) VALUES (4, CAST(N'2023-10-02' AS Date), 1, 4, CAST(600.0000 AS Decimal(9, 4)))
INSERT [dbo].[PortfolioHoldings] ([HID], [AsOfDate], [PortID], [SecID], [HeldShares]) VALUES (5, CAST(N'2023-10-03' AS Date), 1, 1, CAST(1000.0000 AS Decimal(9, 4)))
INSERT [dbo].[PortfolioHoldings] ([HID], [AsOfDate], [PortID], [SecID], [HeldShares]) VALUES (6, CAST(N'2023-10-03' AS Date), 1, 2, CAST(1000.0000 AS Decimal(9, 4)))
INSERT [dbo].[PortfolioHoldings] ([HID], [AsOfDate], [PortID], [SecID], [HeldShares]) VALUES (7, CAST(N'2023-10-03' AS Date), 1, 3, CAST(1000.0000 AS Decimal(9, 4)))
INSERT [dbo].[PortfolioHoldings] ([HID], [AsOfDate], [PortID], [SecID], [HeldShares]) VALUES (9, CAST(N'2023-10-04' AS Date), 1, 1, CAST(1000.0000 AS Decimal(9, 4)))
INSERT [dbo].[PortfolioHoldings] ([HID], [AsOfDate], [PortID], [SecID], [HeldShares]) VALUES (10, CAST(N'2023-10-04' AS Date), 1, 2, CAST(1000.0000 AS Decimal(9, 4)))
INSERT [dbo].[PortfolioHoldings] ([HID], [AsOfDate], [PortID], [SecID], [HeldShares]) VALUES (11, CAST(N'2023-10-04' AS Date), 1, 3, CAST(1000.0000 AS Decimal(9, 4)))
INSERT [dbo].[PortfolioHoldings] ([HID], [AsOfDate], [PortID], [SecID], [HeldShares]) VALUES (12, CAST(N'2023-10-04' AS Date), 1, 4, CAST(1000.0000 AS Decimal(9, 4)))
INSERT [dbo].[PortfolioHoldings] ([HID], [AsOfDate], [PortID], [SecID], [HeldShares]) VALUES (13, CAST(N'2023-10-05' AS Date), 1, 1, CAST(1000.0000 AS Decimal(9, 4)))
INSERT [dbo].[PortfolioHoldings] ([HID], [AsOfDate], [PortID], [SecID], [HeldShares]) VALUES (14, CAST(N'2023-10-05' AS Date), 1, 2, CAST(1000.0000 AS Decimal(9, 4)))
INSERT [dbo].[PortfolioHoldings] ([HID], [AsOfDate], [PortID], [SecID], [HeldShares]) VALUES (15, CAST(N'2023-10-05' AS Date), 1, 3, CAST(1000.0000 AS Decimal(9, 4)))
INSERT [dbo].[PortfolioHoldings] ([HID], [AsOfDate], [PortID], [SecID], [HeldShares]) VALUES (16, CAST(N'2023-10-05' AS Date), 1, 4, CAST(1000.0000 AS Decimal(9, 4)))
INSERT [dbo].[PortfolioHoldings] ([HID], [AsOfDate], [PortID], [SecID], [HeldShares]) VALUES (17, CAST(N'2023-10-06' AS Date), 1, 1, CAST(1000.0000 AS Decimal(9, 4)))
INSERT [dbo].[PortfolioHoldings] ([HID], [AsOfDate], [PortID], [SecID], [HeldShares]) VALUES (18, CAST(N'2023-10-06' AS Date), 1, 2, CAST(1000.0000 AS Decimal(9, 4)))
INSERT [dbo].[PortfolioHoldings] ([HID], [AsOfDate], [PortID], [SecID], [HeldShares]) VALUES (19, CAST(N'2023-10-06' AS Date), 1, 3, CAST(1000.0000 AS Decimal(9, 4)))
INSERT [dbo].[PortfolioHoldings] ([HID], [AsOfDate], [PortID], [SecID], [HeldShares]) VALUES (20, CAST(N'2023-10-06' AS Date), 1, 4, CAST(1000.0000 AS Decimal(9, 4)))
INSERT [dbo].[PortfolioHoldings] ([HID], [AsOfDate], [PortID], [SecID], [HeldShares]) VALUES (21, CAST(N'2023-10-09' AS Date), 1, 1, CAST(1000.0000 AS Decimal(9, 4)))
INSERT [dbo].[PortfolioHoldings] ([HID], [AsOfDate], [PortID], [SecID], [HeldShares]) VALUES (22, CAST(N'2023-10-09' AS Date), 1, 2, CAST(1000.0000 AS Decimal(9, 4)))
INSERT [dbo].[PortfolioHoldings] ([HID], [AsOfDate], [PortID], [SecID], [HeldShares]) VALUES (23, CAST(N'2023-10-09' AS Date), 1, 3, CAST(1000.0000 AS Decimal(9, 4)))
INSERT [dbo].[PortfolioHoldings] ([HID], [AsOfDate], [PortID], [SecID], [HeldShares]) VALUES (24, CAST(N'2023-10-09' AS Date), 1, 4, CAST(1000.0000 AS Decimal(9, 4)))
INSERT [dbo].[PortfolioHoldings] ([HID], [AsOfDate], [PortID], [SecID], [HeldShares]) VALUES (25, CAST(N'2023-10-10' AS Date), 1, 1, CAST(1000.0000 AS Decimal(9, 4)))
INSERT [dbo].[PortfolioHoldings] ([HID], [AsOfDate], [PortID], [SecID], [HeldShares]) VALUES (26, CAST(N'2023-10-10' AS Date), 1, 2, CAST(1000.0000 AS Decimal(9, 4)))
INSERT [dbo].[PortfolioHoldings] ([HID], [AsOfDate], [PortID], [SecID], [HeldShares]) VALUES (27, CAST(N'2023-10-10' AS Date), 1, 3, CAST(1000.0000 AS Decimal(9, 4)))
INSERT [dbo].[PortfolioHoldings] ([HID], [AsOfDate], [PortID], [SecID], [HeldShares]) VALUES (28, CAST(N'2023-10-10' AS Date), 1, 4, CAST(1000.0000 AS Decimal(9, 4)))
INSERT [dbo].[PortfolioHoldings] ([HID], [AsOfDate], [PortID], [SecID], [HeldShares]) VALUES (29, CAST(N'2023-10-11' AS Date), 1, 1, CAST(1000.0000 AS Decimal(9, 4)))
INSERT [dbo].[PortfolioHoldings] ([HID], [AsOfDate], [PortID], [SecID], [HeldShares]) VALUES (30, CAST(N'2023-10-11' AS Date), 1, 2, CAST(1000.0000 AS Decimal(9, 4)))
INSERT [dbo].[PortfolioHoldings] ([HID], [AsOfDate], [PortID], [SecID], [HeldShares]) VALUES (31, CAST(N'2023-10-11' AS Date), 1, 3, CAST(1000.0000 AS Decimal(9, 4)))
INSERT [dbo].[PortfolioHoldings] ([HID], [AsOfDate], [PortID], [SecID], [HeldShares]) VALUES (32, CAST(N'2023-10-11' AS Date), 1, 4, CAST(1000.0000 AS Decimal(9, 4)))
INSERT [dbo].[PortfolioHoldings] ([HID], [AsOfDate], [PortID], [SecID], [HeldShares]) VALUES (33, CAST(N'2023-10-12' AS Date), 1, 1, CAST(1000.0000 AS Decimal(9, 4)))
INSERT [dbo].[PortfolioHoldings] ([HID], [AsOfDate], [PortID], [SecID], [HeldShares]) VALUES (34, CAST(N'2023-10-12' AS Date), 1, 2, CAST(1000.0000 AS Decimal(9, 4)))
INSERT [dbo].[PortfolioHoldings] ([HID], [AsOfDate], [PortID], [SecID], [HeldShares]) VALUES (35, CAST(N'2023-10-12' AS Date), 1, 3, CAST(1000.0000 AS Decimal(9, 4)))
INSERT [dbo].[PortfolioHoldings] ([HID], [AsOfDate], [PortID], [SecID], [HeldShares]) VALUES (36, CAST(N'2023-10-12' AS Date), 1, 4, CAST(1000.0000 AS Decimal(9, 4)))
INSERT [dbo].[PortfolioHoldings] ([HID], [AsOfDate], [PortID], [SecID], [HeldShares]) VALUES (37, CAST(N'2023-10-13' AS Date), 1, 1, CAST(1000.0000 AS Decimal(9, 4)))
INSERT [dbo].[PortfolioHoldings] ([HID], [AsOfDate], [PortID], [SecID], [HeldShares]) VALUES (38, CAST(N'2023-10-13' AS Date), 1, 2, CAST(1000.0000 AS Decimal(9, 4)))
INSERT [dbo].[PortfolioHoldings] ([HID], [AsOfDate], [PortID], [SecID], [HeldShares]) VALUES (39, CAST(N'2023-10-13' AS Date), 1, 3, CAST(1000.0000 AS Decimal(9, 4)))
INSERT [dbo].[PortfolioHoldings] ([HID], [AsOfDate], [PortID], [SecID], [HeldShares]) VALUES (40, CAST(N'2023-10-13' AS Date), 1, 4, CAST(1000.0000 AS Decimal(9, 4)))
INSERT [dbo].[PortfolioHoldings] ([HID], [AsOfDate], [PortID], [SecID], [HeldShares]) VALUES (41, CAST(N'2023-10-16' AS Date), 1, 1, CAST(1000.0000 AS Decimal(9, 4)))
INSERT [dbo].[PortfolioHoldings] ([HID], [AsOfDate], [PortID], [SecID], [HeldShares]) VALUES (42, CAST(N'2023-10-16' AS Date), 1, 2, CAST(1000.0000 AS Decimal(9, 4)))
INSERT [dbo].[PortfolioHoldings] ([HID], [AsOfDate], [PortID], [SecID], [HeldShares]) VALUES (43, CAST(N'2023-10-16' AS Date), 1, 3, CAST(1000.0000 AS Decimal(9, 4)))
INSERT [dbo].[PortfolioHoldings] ([HID], [AsOfDate], [PortID], [SecID], [HeldShares]) VALUES (44, CAST(N'2023-10-16' AS Date), 1, 4, CAST(1000.0000 AS Decimal(9, 4)))
INSERT [dbo].[PortfolioHoldings] ([HID], [AsOfDate], [PortID], [SecID], [HeldShares]) VALUES (45, CAST(N'2023-10-17' AS Date), 1, 1, CAST(1000.0000 AS Decimal(9, 4)))
INSERT [dbo].[PortfolioHoldings] ([HID], [AsOfDate], [PortID], [SecID], [HeldShares]) VALUES (46, CAST(N'2023-10-17' AS Date), 1, 2, CAST(1000.0000 AS Decimal(9, 4)))
INSERT [dbo].[PortfolioHoldings] ([HID], [AsOfDate], [PortID], [SecID], [HeldShares]) VALUES (47, CAST(N'2023-10-17' AS Date), 1, 3, CAST(1000.0000 AS Decimal(9, 4)))
INSERT [dbo].[PortfolioHoldings] ([HID], [AsOfDate], [PortID], [SecID], [HeldShares]) VALUES (48, CAST(N'2023-10-17' AS Date), 1, 4, CAST(1000.0000 AS Decimal(9, 4)))
INSERT [dbo].[PortfolioHoldings] ([HID], [AsOfDate], [PortID], [SecID], [HeldShares]) VALUES (49, CAST(N'2023-10-18' AS Date), 1, 1, CAST(1000.0000 AS Decimal(9, 4)))
INSERT [dbo].[PortfolioHoldings] ([HID], [AsOfDate], [PortID], [SecID], [HeldShares]) VALUES (50, CAST(N'2023-10-18' AS Date), 1, 2, CAST(1000.0000 AS Decimal(9, 4)))
INSERT [dbo].[PortfolioHoldings] ([HID], [AsOfDate], [PortID], [SecID], [HeldShares]) VALUES (51, CAST(N'2023-10-18' AS Date), 1, 3, CAST(1000.0000 AS Decimal(9, 4)))
INSERT [dbo].[PortfolioHoldings] ([HID], [AsOfDate], [PortID], [SecID], [HeldShares]) VALUES (52, CAST(N'2023-10-18' AS Date), 1, 4, CAST(1000.0000 AS Decimal(9, 4)))
INSERT [dbo].[PortfolioHoldings] ([HID], [AsOfDate], [PortID], [SecID], [HeldShares]) VALUES (53, CAST(N'2023-10-19' AS Date), 1, 1, CAST(1000.0000 AS Decimal(9, 4)))
INSERT [dbo].[PortfolioHoldings] ([HID], [AsOfDate], [PortID], [SecID], [HeldShares]) VALUES (54, CAST(N'2023-10-19' AS Date), 1, 2, CAST(1000.0000 AS Decimal(9, 4)))
INSERT [dbo].[PortfolioHoldings] ([HID], [AsOfDate], [PortID], [SecID], [HeldShares]) VALUES (55, CAST(N'2023-10-19' AS Date), 1, 3, CAST(1000.0000 AS Decimal(9, 4)))
INSERT [dbo].[PortfolioHoldings] ([HID], [AsOfDate], [PortID], [SecID], [HeldShares]) VALUES (56, CAST(N'2023-10-19' AS Date), 1, 4, CAST(1000.0000 AS Decimal(9, 4)))
INSERT [dbo].[PortfolioHoldings] ([HID], [AsOfDate], [PortID], [SecID], [HeldShares]) VALUES (57, CAST(N'2023-10-20' AS Date), 1, 1, CAST(1000.0000 AS Decimal(9, 4)))
INSERT [dbo].[PortfolioHoldings] ([HID], [AsOfDate], [PortID], [SecID], [HeldShares]) VALUES (58, CAST(N'2023-10-20' AS Date), 1, 2, CAST(1000.0000 AS Decimal(9, 4)))
INSERT [dbo].[PortfolioHoldings] ([HID], [AsOfDate], [PortID], [SecID], [HeldShares]) VALUES (59, CAST(N'2023-10-20' AS Date), 1, 3, CAST(1000.0000 AS Decimal(9, 4)))
INSERT [dbo].[PortfolioHoldings] ([HID], [AsOfDate], [PortID], [SecID], [HeldShares]) VALUES (60, CAST(N'2023-10-20' AS Date), 1, 4, CAST(1000.0000 AS Decimal(9, 4)))
INSERT [dbo].[PortfolioHoldings] ([HID], [AsOfDate], [PortID], [SecID], [HeldShares]) VALUES (61, CAST(N'2023-10-23' AS Date), 1, 1, CAST(1000.0000 AS Decimal(9, 4)))
INSERT [dbo].[PortfolioHoldings] ([HID], [AsOfDate], [PortID], [SecID], [HeldShares]) VALUES (62, CAST(N'2023-10-23' AS Date), 1, 2, CAST(1000.0000 AS Decimal(9, 4)))
INSERT [dbo].[PortfolioHoldings] ([HID], [AsOfDate], [PortID], [SecID], [HeldShares]) VALUES (63, CAST(N'2023-10-23' AS Date), 1, 3, CAST(1000.0000 AS Decimal(9, 4)))
INSERT [dbo].[PortfolioHoldings] ([HID], [AsOfDate], [PortID], [SecID], [HeldShares]) VALUES (64, CAST(N'2023-10-23' AS Date), 1, 4, CAST(1000.0000 AS Decimal(9, 4)))
INSERT [dbo].[PortfolioHoldings] ([HID], [AsOfDate], [PortID], [SecID], [HeldShares]) VALUES (65, CAST(N'2023-10-24' AS Date), 1, 1, CAST(1000.0000 AS Decimal(9, 4)))
INSERT [dbo].[PortfolioHoldings] ([HID], [AsOfDate], [PortID], [SecID], [HeldShares]) VALUES (66, CAST(N'2023-10-24' AS Date), 1, 2, CAST(1000.0000 AS Decimal(9, 4)))
INSERT [dbo].[PortfolioHoldings] ([HID], [AsOfDate], [PortID], [SecID], [HeldShares]) VALUES (67, CAST(N'2023-10-24' AS Date), 1, 3, CAST(1000.0000 AS Decimal(9, 4)))
INSERT [dbo].[PortfolioHoldings] ([HID], [AsOfDate], [PortID], [SecID], [HeldShares]) VALUES (68, CAST(N'2023-10-24' AS Date), 1, 4, CAST(1000.0000 AS Decimal(9, 4)))
INSERT [dbo].[PortfolioHoldings] ([HID], [AsOfDate], [PortID], [SecID], [HeldShares]) VALUES (69, CAST(N'2023-10-25' AS Date), 1, 1, CAST(1000.0000 AS Decimal(9, 4)))
INSERT [dbo].[PortfolioHoldings] ([HID], [AsOfDate], [PortID], [SecID], [HeldShares]) VALUES (70, CAST(N'2023-10-25' AS Date), 1, 2, CAST(1000.0000 AS Decimal(9, 4)))
INSERT [dbo].[PortfolioHoldings] ([HID], [AsOfDate], [PortID], [SecID], [HeldShares]) VALUES (71, CAST(N'2023-10-25' AS Date), 1, 3, CAST(1000.0000 AS Decimal(9, 4)))
INSERT [dbo].[PortfolioHoldings] ([HID], [AsOfDate], [PortID], [SecID], [HeldShares]) VALUES (72, CAST(N'2023-10-25' AS Date), 1, 4, CAST(1000.0000 AS Decimal(9, 4)))
INSERT [dbo].[PortfolioHoldings] ([HID], [AsOfDate], [PortID], [SecID], [HeldShares]) VALUES (73, CAST(N'2023-10-26' AS Date), 1, 1, CAST(1000.0000 AS Decimal(9, 4)))
INSERT [dbo].[PortfolioHoldings] ([HID], [AsOfDate], [PortID], [SecID], [HeldShares]) VALUES (74, CAST(N'2023-10-26' AS Date), 1, 2, CAST(1000.0000 AS Decimal(9, 4)))
INSERT [dbo].[PortfolioHoldings] ([HID], [AsOfDate], [PortID], [SecID], [HeldShares]) VALUES (75, CAST(N'2023-10-26' AS Date), 1, 3, CAST(1000.0000 AS Decimal(9, 4)))
INSERT [dbo].[PortfolioHoldings] ([HID], [AsOfDate], [PortID], [SecID], [HeldShares]) VALUES (76, CAST(N'2023-10-26' AS Date), 1, 4, CAST(1000.0000 AS Decimal(9, 4)))
INSERT [dbo].[PortfolioHoldings] ([HID], [AsOfDate], [PortID], [SecID], [HeldShares]) VALUES (77, CAST(N'2023-10-27' AS Date), 1, 1, CAST(1000.0000 AS Decimal(9, 4)))
INSERT [dbo].[PortfolioHoldings] ([HID], [AsOfDate], [PortID], [SecID], [HeldShares]) VALUES (78, CAST(N'2023-10-27' AS Date), 1, 2, CAST(1000.0000 AS Decimal(9, 4)))
INSERT [dbo].[PortfolioHoldings] ([HID], [AsOfDate], [PortID], [SecID], [HeldShares]) VALUES (79, CAST(N'2023-10-27' AS Date), 1, 3, CAST(1000.0000 AS Decimal(9, 4)))
INSERT [dbo].[PortfolioHoldings] ([HID], [AsOfDate], [PortID], [SecID], [HeldShares]) VALUES (80, CAST(N'2023-10-27' AS Date), 1, 4, CAST(1000.0000 AS Decimal(9, 4)))
INSERT [dbo].[PortfolioHoldings] ([HID], [AsOfDate], [PortID], [SecID], [HeldShares]) VALUES (81, CAST(N'2023-10-30' AS Date), 1, 1, CAST(1000.0000 AS Decimal(9, 4)))
INSERT [dbo].[PortfolioHoldings] ([HID], [AsOfDate], [PortID], [SecID], [HeldShares]) VALUES (82, CAST(N'2023-10-30' AS Date), 1, 2, CAST(1000.0000 AS Decimal(9, 4)))
INSERT [dbo].[PortfolioHoldings] ([HID], [AsOfDate], [PortID], [SecID], [HeldShares]) VALUES (83, CAST(N'2023-10-30' AS Date), 1, 3, CAST(1000.0000 AS Decimal(9, 4)))
INSERT [dbo].[PortfolioHoldings] ([HID], [AsOfDate], [PortID], [SecID], [HeldShares]) VALUES (84, CAST(N'2023-10-30' AS Date), 1, 4, CAST(1000.0000 AS Decimal(9, 4)))
INSERT [dbo].[PortfolioHoldings] ([HID], [AsOfDate], [PortID], [SecID], [HeldShares]) VALUES (85, CAST(N'2023-10-02' AS Date), 2, 22, CAST(1000.0000 AS Decimal(9, 4)))
INSERT [dbo].[PortfolioHoldings] ([HID], [AsOfDate], [PortID], [SecID], [HeldShares]) VALUES (86, CAST(N'2023-10-02' AS Date), 2, 23, CAST(1000.0000 AS Decimal(9, 4)))
INSERT [dbo].[PortfolioHoldings] ([HID], [AsOfDate], [PortID], [SecID], [HeldShares]) VALUES (87, CAST(N'2023-10-02' AS Date), 2, 24, CAST(1000.0000 AS Decimal(9, 4)))
INSERT [dbo].[PortfolioHoldings] ([HID], [AsOfDate], [PortID], [SecID], [HeldShares]) VALUES (88, CAST(N'2023-10-02' AS Date), 2, 25, CAST(1000.0000 AS Decimal(9, 4)))
INSERT [dbo].[PortfolioHoldings] ([HID], [AsOfDate], [PortID], [SecID], [HeldShares]) VALUES (89, CAST(N'2023-10-03' AS Date), 2, 22, CAST(1000.0000 AS Decimal(9, 4)))
INSERT [dbo].[PortfolioHoldings] ([HID], [AsOfDate], [PortID], [SecID], [HeldShares]) VALUES (90, CAST(N'2023-10-03' AS Date), 2, 23, CAST(1000.0000 AS Decimal(9, 4)))
INSERT [dbo].[PortfolioHoldings] ([HID], [AsOfDate], [PortID], [SecID], [HeldShares]) VALUES (91, CAST(N'2023-10-03' AS Date), 2, 24, CAST(1000.0000 AS Decimal(9, 4)))
INSERT [dbo].[PortfolioHoldings] ([HID], [AsOfDate], [PortID], [SecID], [HeldShares]) VALUES (92, CAST(N'2023-10-03' AS Date), 2, 25, CAST(1000.0000 AS Decimal(9, 4)))
INSERT [dbo].[PortfolioHoldings] ([HID], [AsOfDate], [PortID], [SecID], [HeldShares]) VALUES (93, CAST(N'2023-10-04' AS Date), 2, 22, CAST(1000.0000 AS Decimal(9, 4)))
INSERT [dbo].[PortfolioHoldings] ([HID], [AsOfDate], [PortID], [SecID], [HeldShares]) VALUES (94, CAST(N'2023-10-04' AS Date), 2, 23, CAST(1000.0000 AS Decimal(9, 4)))
INSERT [dbo].[PortfolioHoldings] ([HID], [AsOfDate], [PortID], [SecID], [HeldShares]) VALUES (95, CAST(N'2023-10-04' AS Date), 2, 24, CAST(1000.0000 AS Decimal(9, 4)))
INSERT [dbo].[PortfolioHoldings] ([HID], [AsOfDate], [PortID], [SecID], [HeldShares]) VALUES (96, CAST(N'2023-10-04' AS Date), 2, 25, CAST(1000.0000 AS Decimal(9, 4)))
INSERT [dbo].[PortfolioHoldings] ([HID], [AsOfDate], [PortID], [SecID], [HeldShares]) VALUES (97, CAST(N'2023-10-05' AS Date), 2, 22, CAST(1000.0000 AS Decimal(9, 4)))
INSERT [dbo].[PortfolioHoldings] ([HID], [AsOfDate], [PortID], [SecID], [HeldShares]) VALUES (98, CAST(N'2023-10-05' AS Date), 2, 23, CAST(1000.0000 AS Decimal(9, 4)))
INSERT [dbo].[PortfolioHoldings] ([HID], [AsOfDate], [PortID], [SecID], [HeldShares]) VALUES (99, CAST(N'2023-10-05' AS Date), 2, 24, CAST(1000.0000 AS Decimal(9, 4)))
INSERT [dbo].[PortfolioHoldings] ([HID], [AsOfDate], [PortID], [SecID], [HeldShares]) VALUES (100, CAST(N'2023-10-05' AS Date), 2, 25, CAST(1000.0000 AS Decimal(9, 4)))
GO
INSERT [dbo].[PortfolioHoldings] ([HID], [AsOfDate], [PortID], [SecID], [HeldShares]) VALUES (101, CAST(N'2023-10-06' AS Date), 2, 22, CAST(1000.0000 AS Decimal(9, 4)))
INSERT [dbo].[PortfolioHoldings] ([HID], [AsOfDate], [PortID], [SecID], [HeldShares]) VALUES (102, CAST(N'2023-10-06' AS Date), 2, 23, CAST(1000.0000 AS Decimal(9, 4)))
INSERT [dbo].[PortfolioHoldings] ([HID], [AsOfDate], [PortID], [SecID], [HeldShares]) VALUES (103, CAST(N'2023-10-06' AS Date), 2, 24, CAST(1000.0000 AS Decimal(9, 4)))
INSERT [dbo].[PortfolioHoldings] ([HID], [AsOfDate], [PortID], [SecID], [HeldShares]) VALUES (104, CAST(N'2023-10-06' AS Date), 2, 25, CAST(1000.0000 AS Decimal(9, 4)))
INSERT [dbo].[PortfolioHoldings] ([HID], [AsOfDate], [PortID], [SecID], [HeldShares]) VALUES (105, CAST(N'2023-10-09' AS Date), 2, 22, CAST(1000.0000 AS Decimal(9, 4)))
INSERT [dbo].[PortfolioHoldings] ([HID], [AsOfDate], [PortID], [SecID], [HeldShares]) VALUES (106, CAST(N'2023-10-09' AS Date), 2, 23, CAST(1000.0000 AS Decimal(9, 4)))
INSERT [dbo].[PortfolioHoldings] ([HID], [AsOfDate], [PortID], [SecID], [HeldShares]) VALUES (107, CAST(N'2023-10-09' AS Date), 2, 24, CAST(1000.0000 AS Decimal(9, 4)))
INSERT [dbo].[PortfolioHoldings] ([HID], [AsOfDate], [PortID], [SecID], [HeldShares]) VALUES (108, CAST(N'2023-10-09' AS Date), 2, 25, CAST(1000.0000 AS Decimal(9, 4)))
INSERT [dbo].[PortfolioHoldings] ([HID], [AsOfDate], [PortID], [SecID], [HeldShares]) VALUES (109, CAST(N'2023-10-10' AS Date), 2, 22, CAST(1000.0000 AS Decimal(9, 4)))
INSERT [dbo].[PortfolioHoldings] ([HID], [AsOfDate], [PortID], [SecID], [HeldShares]) VALUES (110, CAST(N'2023-10-10' AS Date), 2, 23, CAST(1000.0000 AS Decimal(9, 4)))
INSERT [dbo].[PortfolioHoldings] ([HID], [AsOfDate], [PortID], [SecID], [HeldShares]) VALUES (111, CAST(N'2023-10-10' AS Date), 2, 24, CAST(1000.0000 AS Decimal(9, 4)))
INSERT [dbo].[PortfolioHoldings] ([HID], [AsOfDate], [PortID], [SecID], [HeldShares]) VALUES (112, CAST(N'2023-10-10' AS Date), 2, 25, CAST(1000.0000 AS Decimal(9, 4)))
INSERT [dbo].[PortfolioHoldings] ([HID], [AsOfDate], [PortID], [SecID], [HeldShares]) VALUES (113, CAST(N'2023-10-11' AS Date), 2, 22, CAST(1000.0000 AS Decimal(9, 4)))
INSERT [dbo].[PortfolioHoldings] ([HID], [AsOfDate], [PortID], [SecID], [HeldShares]) VALUES (114, CAST(N'2023-10-11' AS Date), 2, 23, CAST(1000.0000 AS Decimal(9, 4)))
INSERT [dbo].[PortfolioHoldings] ([HID], [AsOfDate], [PortID], [SecID], [HeldShares]) VALUES (115, CAST(N'2023-10-11' AS Date), 2, 24, CAST(1000.0000 AS Decimal(9, 4)))
INSERT [dbo].[PortfolioHoldings] ([HID], [AsOfDate], [PortID], [SecID], [HeldShares]) VALUES (116, CAST(N'2023-10-11' AS Date), 2, 25, CAST(1000.0000 AS Decimal(9, 4)))
INSERT [dbo].[PortfolioHoldings] ([HID], [AsOfDate], [PortID], [SecID], [HeldShares]) VALUES (117, CAST(N'2023-10-12' AS Date), 2, 22, CAST(1000.0000 AS Decimal(9, 4)))
INSERT [dbo].[PortfolioHoldings] ([HID], [AsOfDate], [PortID], [SecID], [HeldShares]) VALUES (118, CAST(N'2023-10-12' AS Date), 2, 23, CAST(1000.0000 AS Decimal(9, 4)))
INSERT [dbo].[PortfolioHoldings] ([HID], [AsOfDate], [PortID], [SecID], [HeldShares]) VALUES (119, CAST(N'2023-10-12' AS Date), 2, 24, CAST(1000.0000 AS Decimal(9, 4)))
INSERT [dbo].[PortfolioHoldings] ([HID], [AsOfDate], [PortID], [SecID], [HeldShares]) VALUES (120, CAST(N'2023-10-12' AS Date), 2, 25, CAST(1000.0000 AS Decimal(9, 4)))
INSERT [dbo].[PortfolioHoldings] ([HID], [AsOfDate], [PortID], [SecID], [HeldShares]) VALUES (121, CAST(N'2023-10-13' AS Date), 2, 22, CAST(1000.0000 AS Decimal(9, 4)))
INSERT [dbo].[PortfolioHoldings] ([HID], [AsOfDate], [PortID], [SecID], [HeldShares]) VALUES (122, CAST(N'2023-10-13' AS Date), 2, 23, CAST(1000.0000 AS Decimal(9, 4)))
INSERT [dbo].[PortfolioHoldings] ([HID], [AsOfDate], [PortID], [SecID], [HeldShares]) VALUES (123, CAST(N'2023-10-13' AS Date), 2, 24, CAST(1000.0000 AS Decimal(9, 4)))
INSERT [dbo].[PortfolioHoldings] ([HID], [AsOfDate], [PortID], [SecID], [HeldShares]) VALUES (124, CAST(N'2023-10-13' AS Date), 2, 25, CAST(1000.0000 AS Decimal(9, 4)))
INSERT [dbo].[PortfolioHoldings] ([HID], [AsOfDate], [PortID], [SecID], [HeldShares]) VALUES (125, CAST(N'2023-10-16' AS Date), 2, 22, CAST(1000.0000 AS Decimal(9, 4)))
INSERT [dbo].[PortfolioHoldings] ([HID], [AsOfDate], [PortID], [SecID], [HeldShares]) VALUES (126, CAST(N'2023-10-16' AS Date), 2, 23, CAST(1000.0000 AS Decimal(9, 4)))
INSERT [dbo].[PortfolioHoldings] ([HID], [AsOfDate], [PortID], [SecID], [HeldShares]) VALUES (127, CAST(N'2023-10-16' AS Date), 2, 24, CAST(1000.0000 AS Decimal(9, 4)))
INSERT [dbo].[PortfolioHoldings] ([HID], [AsOfDate], [PortID], [SecID], [HeldShares]) VALUES (128, CAST(N'2023-10-16' AS Date), 2, 25, CAST(1000.0000 AS Decimal(9, 4)))
INSERT [dbo].[PortfolioHoldings] ([HID], [AsOfDate], [PortID], [SecID], [HeldShares]) VALUES (129, CAST(N'2023-10-17' AS Date), 2, 22, CAST(1000.0000 AS Decimal(9, 4)))
INSERT [dbo].[PortfolioHoldings] ([HID], [AsOfDate], [PortID], [SecID], [HeldShares]) VALUES (130, CAST(N'2023-10-17' AS Date), 2, 23, CAST(1000.0000 AS Decimal(9, 4)))
INSERT [dbo].[PortfolioHoldings] ([HID], [AsOfDate], [PortID], [SecID], [HeldShares]) VALUES (131, CAST(N'2023-10-17' AS Date), 2, 24, CAST(1000.0000 AS Decimal(9, 4)))
INSERT [dbo].[PortfolioHoldings] ([HID], [AsOfDate], [PortID], [SecID], [HeldShares]) VALUES (132, CAST(N'2023-10-17' AS Date), 2, 25, CAST(1000.0000 AS Decimal(9, 4)))
INSERT [dbo].[PortfolioHoldings] ([HID], [AsOfDate], [PortID], [SecID], [HeldShares]) VALUES (133, CAST(N'2023-10-18' AS Date), 2, 22, CAST(1000.0000 AS Decimal(9, 4)))
INSERT [dbo].[PortfolioHoldings] ([HID], [AsOfDate], [PortID], [SecID], [HeldShares]) VALUES (134, CAST(N'2023-10-18' AS Date), 2, 23, CAST(1000.0000 AS Decimal(9, 4)))
INSERT [dbo].[PortfolioHoldings] ([HID], [AsOfDate], [PortID], [SecID], [HeldShares]) VALUES (135, CAST(N'2023-10-18' AS Date), 2, 24, CAST(1000.0000 AS Decimal(9, 4)))
INSERT [dbo].[PortfolioHoldings] ([HID], [AsOfDate], [PortID], [SecID], [HeldShares]) VALUES (136, CAST(N'2023-10-18' AS Date), 2, 25, CAST(1000.0000 AS Decimal(9, 4)))
INSERT [dbo].[PortfolioHoldings] ([HID], [AsOfDate], [PortID], [SecID], [HeldShares]) VALUES (137, CAST(N'2023-10-19' AS Date), 2, 22, CAST(1000.0000 AS Decimal(9, 4)))
INSERT [dbo].[PortfolioHoldings] ([HID], [AsOfDate], [PortID], [SecID], [HeldShares]) VALUES (138, CAST(N'2023-10-19' AS Date), 2, 23, CAST(1000.0000 AS Decimal(9, 4)))
INSERT [dbo].[PortfolioHoldings] ([HID], [AsOfDate], [PortID], [SecID], [HeldShares]) VALUES (139, CAST(N'2023-10-19' AS Date), 2, 24, CAST(1000.0000 AS Decimal(9, 4)))
INSERT [dbo].[PortfolioHoldings] ([HID], [AsOfDate], [PortID], [SecID], [HeldShares]) VALUES (140, CAST(N'2023-10-19' AS Date), 2, 25, CAST(1000.0000 AS Decimal(9, 4)))
INSERT [dbo].[PortfolioHoldings] ([HID], [AsOfDate], [PortID], [SecID], [HeldShares]) VALUES (141, CAST(N'2023-10-20' AS Date), 2, 22, CAST(1000.0000 AS Decimal(9, 4)))
INSERT [dbo].[PortfolioHoldings] ([HID], [AsOfDate], [PortID], [SecID], [HeldShares]) VALUES (142, CAST(N'2023-10-20' AS Date), 2, 23, CAST(1000.0000 AS Decimal(9, 4)))
INSERT [dbo].[PortfolioHoldings] ([HID], [AsOfDate], [PortID], [SecID], [HeldShares]) VALUES (143, CAST(N'2023-10-20' AS Date), 2, 24, CAST(1000.0000 AS Decimal(9, 4)))
INSERT [dbo].[PortfolioHoldings] ([HID], [AsOfDate], [PortID], [SecID], [HeldShares]) VALUES (144, CAST(N'2023-10-20' AS Date), 2, 25, CAST(1000.0000 AS Decimal(9, 4)))
INSERT [dbo].[PortfolioHoldings] ([HID], [AsOfDate], [PortID], [SecID], [HeldShares]) VALUES (145, CAST(N'2023-10-23' AS Date), 2, 22, CAST(1000.0000 AS Decimal(9, 4)))
INSERT [dbo].[PortfolioHoldings] ([HID], [AsOfDate], [PortID], [SecID], [HeldShares]) VALUES (146, CAST(N'2023-10-23' AS Date), 2, 23, CAST(1000.0000 AS Decimal(9, 4)))
INSERT [dbo].[PortfolioHoldings] ([HID], [AsOfDate], [PortID], [SecID], [HeldShares]) VALUES (147, CAST(N'2023-10-23' AS Date), 2, 24, CAST(1000.0000 AS Decimal(9, 4)))
INSERT [dbo].[PortfolioHoldings] ([HID], [AsOfDate], [PortID], [SecID], [HeldShares]) VALUES (148, CAST(N'2023-10-23' AS Date), 2, 25, CAST(1000.0000 AS Decimal(9, 4)))
INSERT [dbo].[PortfolioHoldings] ([HID], [AsOfDate], [PortID], [SecID], [HeldShares]) VALUES (149, CAST(N'2023-10-24' AS Date), 2, 22, CAST(1000.0000 AS Decimal(9, 4)))
INSERT [dbo].[PortfolioHoldings] ([HID], [AsOfDate], [PortID], [SecID], [HeldShares]) VALUES (150, CAST(N'2023-10-24' AS Date), 2, 23, CAST(1000.0000 AS Decimal(9, 4)))
INSERT [dbo].[PortfolioHoldings] ([HID], [AsOfDate], [PortID], [SecID], [HeldShares]) VALUES (151, CAST(N'2023-10-24' AS Date), 2, 24, CAST(1000.0000 AS Decimal(9, 4)))
INSERT [dbo].[PortfolioHoldings] ([HID], [AsOfDate], [PortID], [SecID], [HeldShares]) VALUES (152, CAST(N'2023-10-24' AS Date), 2, 25, CAST(1000.0000 AS Decimal(9, 4)))
INSERT [dbo].[PortfolioHoldings] ([HID], [AsOfDate], [PortID], [SecID], [HeldShares]) VALUES (153, CAST(N'2023-10-25' AS Date), 2, 22, CAST(1000.0000 AS Decimal(9, 4)))
INSERT [dbo].[PortfolioHoldings] ([HID], [AsOfDate], [PortID], [SecID], [HeldShares]) VALUES (154, CAST(N'2023-10-25' AS Date), 2, 23, CAST(1000.0000 AS Decimal(9, 4)))
INSERT [dbo].[PortfolioHoldings] ([HID], [AsOfDate], [PortID], [SecID], [HeldShares]) VALUES (155, CAST(N'2023-10-25' AS Date), 2, 24, CAST(1000.0000 AS Decimal(9, 4)))
INSERT [dbo].[PortfolioHoldings] ([HID], [AsOfDate], [PortID], [SecID], [HeldShares]) VALUES (156, CAST(N'2023-10-25' AS Date), 2, 25, CAST(1000.0000 AS Decimal(9, 4)))
INSERT [dbo].[PortfolioHoldings] ([HID], [AsOfDate], [PortID], [SecID], [HeldShares]) VALUES (157, CAST(N'2023-10-26' AS Date), 2, 22, CAST(1000.0000 AS Decimal(9, 4)))
INSERT [dbo].[PortfolioHoldings] ([HID], [AsOfDate], [PortID], [SecID], [HeldShares]) VALUES (158, CAST(N'2023-10-26' AS Date), 2, 23, CAST(1000.0000 AS Decimal(9, 4)))
INSERT [dbo].[PortfolioHoldings] ([HID], [AsOfDate], [PortID], [SecID], [HeldShares]) VALUES (159, CAST(N'2023-10-26' AS Date), 2, 24, CAST(1000.0000 AS Decimal(9, 4)))
INSERT [dbo].[PortfolioHoldings] ([HID], [AsOfDate], [PortID], [SecID], [HeldShares]) VALUES (160, CAST(N'2023-10-26' AS Date), 2, 25, CAST(1000.0000 AS Decimal(9, 4)))
INSERT [dbo].[PortfolioHoldings] ([HID], [AsOfDate], [PortID], [SecID], [HeldShares]) VALUES (161, CAST(N'2023-10-27' AS Date), 2, 22, CAST(1000.0000 AS Decimal(9, 4)))
INSERT [dbo].[PortfolioHoldings] ([HID], [AsOfDate], [PortID], [SecID], [HeldShares]) VALUES (162, CAST(N'2023-10-27' AS Date), 2, 23, CAST(1000.0000 AS Decimal(9, 4)))
INSERT [dbo].[PortfolioHoldings] ([HID], [AsOfDate], [PortID], [SecID], [HeldShares]) VALUES (163, CAST(N'2023-10-27' AS Date), 2, 24, CAST(1000.0000 AS Decimal(9, 4)))
INSERT [dbo].[PortfolioHoldings] ([HID], [AsOfDate], [PortID], [SecID], [HeldShares]) VALUES (164, CAST(N'2023-10-27' AS Date), 2, 25, CAST(1000.0000 AS Decimal(9, 4)))
INSERT [dbo].[PortfolioHoldings] ([HID], [AsOfDate], [PortID], [SecID], [HeldShares]) VALUES (165, CAST(N'2023-10-30' AS Date), 2, 22, CAST(1000.0000 AS Decimal(9, 4)))
INSERT [dbo].[PortfolioHoldings] ([HID], [AsOfDate], [PortID], [SecID], [HeldShares]) VALUES (166, CAST(N'2023-10-30' AS Date), 2, 23, CAST(1000.0000 AS Decimal(9, 4)))
INSERT [dbo].[PortfolioHoldings] ([HID], [AsOfDate], [PortID], [SecID], [HeldShares]) VALUES (167, CAST(N'2023-10-30' AS Date), 2, 24, CAST(1000.0000 AS Decimal(9, 4)))
INSERT [dbo].[PortfolioHoldings] ([HID], [AsOfDate], [PortID], [SecID], [HeldShares]) VALUES (168, CAST(N'2023-10-30' AS Date), 2, 25, CAST(1000.0000 AS Decimal(9, 4)))
SET IDENTITY_INSERT [dbo].[PortfolioHoldings] OFF
GO
SET IDENTITY_INSERT [dbo].[SecurityMaster] ON 

INSERT [dbo].[SecurityMaster] ([SecID], [Ticker], [Sector], [Country], [SecurityType], [SecurityName]) VALUES (1, N'MSFT', N'Technology', N'US', N'Equity', N'Microsoft Corp')
INSERT [dbo].[SecurityMaster] ([SecID], [Ticker], [Sector], [Country], [SecurityType], [SecurityName]) VALUES (2, N'GOOGL', N'Technology', N'US', N'Equity', N'Alphabet Inc')
INSERT [dbo].[SecurityMaster] ([SecID], [Ticker], [Sector], [Country], [SecurityType], [SecurityName]) VALUES (3, N'META', N'Technology', N'US', N'Equity', N'Meta Platforms Inc')
INSERT [dbo].[SecurityMaster] ([SecID], [Ticker], [Sector], [Country], [SecurityType], [SecurityName]) VALUES (4, N'NVDA', N'Technology', N'US', N'Equity', N'NVIDIA Corp')
INSERT [dbo].[SecurityMaster] ([SecID], [Ticker], [Sector], [Country], [SecurityType], [SecurityName]) VALUES (5, N'2330.TW', N'Semiconductors', N'Taiwan', N'Equity', N'TSMC')
INSERT [dbo].[SecurityMaster] ([SecID], [Ticker], [Sector], [Country], [SecurityType], [SecurityName]) VALUES (6, N'AMZN', N'Ecommerce', N'US', N'Equity', N'Amazon.com Inc')
INSERT [dbo].[SecurityMaster] ([SecID], [Ticker], [Sector], [Country], [SecurityType], [SecurityName]) VALUES (7, N'V', N'Payments', N'US', N'Equity', N'Visa Inc')
INSERT [dbo].[SecurityMaster] ([SecID], [Ticker], [Sector], [Country], [SecurityType], [SecurityName]) VALUES (8, N'ASM', N'Semiconductors', N'Europe', N'Equity', N'ASML Holding NV')
INSERT [dbo].[SecurityMaster] ([SecID], [Ticker], [Sector], [Country], [SecurityType], [SecurityName]) VALUES (9, N'NOW', N'Technology', N'US', N'Equity', N'ServiceNow Inc')
INSERT [dbo].[SecurityMaster] ([SecID], [Ticker], [Sector], [Country], [SecurityType], [SecurityName]) VALUES (10, N'CRM', N'Technology', N'US', N'Equity', N'Salesforce Inc')
INSERT [dbo].[SecurityMaster] ([SecID], [Ticker], [Sector], [Country], [SecurityType], [SecurityName]) VALUES (11, N'PCTY', N'Technology', N'US', N'Equity', N'Paylocity Holding Corp')
INSERT [dbo].[SecurityMaster] ([SecID], [Ticker], [Sector], [Country], [SecurityType], [SecurityName]) VALUES (12, N'CAP.PA', N'Technology', N'Europe', N'Equity', N'Capgemini SE')
INSERT [dbo].[SecurityMaster] ([SecID], [Ticker], [Sector], [Country], [SecurityType], [SecurityName]) VALUES (13, N'SNPS', N'Technology', N'US', N'Equity', N'Synopsys Inc')
INSERT [dbo].[SecurityMaster] ([SecID], [Ticker], [Sector], [Country], [SecurityType], [SecurityName]) VALUES (14, N'LRCX', N'Semiconductors', N'US', N'Equity', N'Lam Research Corp')
INSERT [dbo].[SecurityMaster] ([SecID], [Ticker], [Sector], [Country], [SecurityType], [SecurityName]) VALUES (15, N'ADYEN.AS', N'Ecommerce', N'Europe', N'Equity', N'Adyen NV')
INSERT [dbo].[SecurityMaster] ([SecID], [Ticker], [Sector], [Country], [SecurityType], [SecurityName]) VALUES (16, N'ADSK', N'Technology', N'US', N'Equity', N'Autodesk Inc')
INSERT [dbo].[SecurityMaster] ([SecID], [Ticker], [Sector], [Country], [SecurityType], [SecurityName]) VALUES (17, N'MU', N'Technology', N'US', N'Equity', N'Micron Technology Inc')
INSERT [dbo].[SecurityMaster] ([SecID], [Ticker], [Sector], [Country], [SecurityType], [SecurityName]) VALUES (18, N'ASM', N'Semiconductors', N'Europe', N'Equity', N'ASM International NV')
INSERT [dbo].[SecurityMaster] ([SecID], [Ticker], [Sector], [Country], [SecurityType], [SecurityName]) VALUES (19, N'000660.KS', N'Semiconductors', N'South Korea', N'Equity', N'SK Hynix Inc')
INSERT [dbo].[SecurityMaster] ([SecID], [Ticker], [Sector], [Country], [SecurityType], [SecurityName]) VALUES (20, N'ADI', N'Semiconductors', N'US', N'Equity', N'Analog Devices Inc')
INSERT [dbo].[SecurityMaster] ([SecID], [Ticker], [Sector], [Country], [SecurityType], [SecurityName]) VALUES (21, N'IFX.DE', N'Semicondutors', N'Europe', N'Equity', N'Infineon Technologies AG')
INSERT [dbo].[SecurityMaster] ([SecID], [Ticker], [Sector], [Country], [SecurityType], [SecurityName]) VALUES (22, N'DDOG', N'Technology', N'US', N'Equity', N'Datadog Inc')
INSERT [dbo].[SecurityMaster] ([SecID], [Ticker], [Sector], [Country], [SecurityType], [SecurityName]) VALUES (23, N'NET', N'Technology', N'US', N'Equity', N'CLOUDFARE INC')
INSERT [dbo].[SecurityMaster] ([SecID], [Ticker], [Sector], [Country], [SecurityType], [SecurityName]) VALUES (24, N'QCOM', N'Semiconductors', N'US', N'Equity', N'QUALCOMM Inc')
INSERT [dbo].[SecurityMaster] ([SecID], [Ticker], [Sector], [Country], [SecurityType], [SecurityName]) VALUES (25, N'VRNS', N'Technology', N'US', N'Equity', N'Varonis Systems Inc')
INSERT [dbo].[SecurityMaster] ([SecID], [Ticker], [Sector], [Country], [SecurityType], [SecurityName]) VALUES (26, N'035420.KS', N'Technology', N'South Korea', N'Equity', N'NAVER Corp')
INSERT [dbo].[SecurityMaster] ([SecID], [Ticker], [Sector], [Country], [SecurityType], [SecurityName]) VALUES (27, N'TTWO', N'Technology', N'US', N'Equity', N'Take-Two Interactive Software Inc')
INSERT [dbo].[SecurityMaster] ([SecID], [Ticker], [Sector], [Country], [SecurityType], [SecurityName]) VALUES (28, N'TEMN.SW', N'Technology', N'Europe', N'Equity', N'Temenos AG')
INSERT [dbo].[SecurityMaster] ([SecID], [Ticker], [Sector], [Country], [SecurityType], [SecurityName]) VALUES (29, N'AMS', N'Technology', N'Europe', N'Equity', N'Amadeus IT Group SA')
INSERT [dbo].[SecurityMaster] ([SecID], [Ticker], [Sector], [Country], [SecurityType], [SecurityName]) VALUES (30, N'ENTG', N'Semiconductors', N'US', N'Equity', N'Entegris Inc')
INSERT [dbo].[SecurityMaster] ([SecID], [Ticker], [Sector], [Country], [SecurityType], [SecurityName]) VALUES (31, N'OKTA', N'Technology', N'US', N'Equity', N'Okta Inc')
INSERT [dbo].[SecurityMaster] ([SecID], [Ticker], [Sector], [Country], [SecurityType], [SecurityName]) VALUES (32, N'EPAM', N'Technology', N'US', N'Equity', N'EPAM Systems Inc')
INSERT [dbo].[SecurityMaster] ([SecID], [Ticker], [Sector], [Country], [SecurityType], [SecurityName]) VALUES (33, N'ABNB', N'', N'US', N'Equity', N'Airbnb Inc')
INSERT [dbo].[SecurityMaster] ([SecID], [Ticker], [Sector], [Country], [SecurityType], [SecurityName]) VALUES (34, N'ADBE', N'', N'US', N'Equity', N'Adobe Inc')
INSERT [dbo].[SecurityMaster] ([SecID], [Ticker], [Sector], [Country], [SecurityType], [SecurityName]) VALUES (35, N'ADP', N'', N'US', N'Equity', N'Automatic Data Processing Inc')
INSERT [dbo].[SecurityMaster] ([SecID], [Ticker], [Sector], [Country], [SecurityType], [SecurityName]) VALUES (36, N'AEP', N'', N'US', N'Equity', N'American Electric Power Co Inc')
INSERT [dbo].[SecurityMaster] ([SecID], [Ticker], [Sector], [Country], [SecurityType], [SecurityName]) VALUES (37, N'ALGN', N'', N'US', N'Equity', N'Align Technology Inc')
INSERT [dbo].[SecurityMaster] ([SecID], [Ticker], [Sector], [Country], [SecurityType], [SecurityName]) VALUES (38, N'AMAT', N'', N'US', N'Equity', N'Applied Materials Inc')
INSERT [dbo].[SecurityMaster] ([SecID], [Ticker], [Sector], [Country], [SecurityType], [SecurityName]) VALUES (39, N'AMD', N'', N'US', N'Equity', N'Advanced Micro Devices Inc')
INSERT [dbo].[SecurityMaster] ([SecID], [Ticker], [Sector], [Country], [SecurityType], [SecurityName]) VALUES (40, N'AMGN', N'', N'US', N'Equity', N'Amgen Inc')
INSERT [dbo].[SecurityMaster] ([SecID], [Ticker], [Sector], [Country], [SecurityType], [SecurityName]) VALUES (41, N'ANSS', N'', N'US', N'Equity', N'ANSYS Inc')
INSERT [dbo].[SecurityMaster] ([SecID], [Ticker], [Sector], [Country], [SecurityType], [SecurityName]) VALUES (42, N'AVGO', N'', N'US', N'Equity', N'Broadcom Inc')
INSERT [dbo].[SecurityMaster] ([SecID], [Ticker], [Sector], [Country], [SecurityType], [SecurityName]) VALUES (43, N'AZN', N'', N'US', N'Equity', N'AstraZeneca PLC ADR')
INSERT [dbo].[SecurityMaster] ([SecID], [Ticker], [Sector], [Country], [SecurityType], [SecurityName]) VALUES (44, N'BIIB', N'', N'US', N'Equity', N'Biogen Inc')
INSERT [dbo].[SecurityMaster] ([SecID], [Ticker], [Sector], [Country], [SecurityType], [SecurityName]) VALUES (45, N'BKNG', N'', N'US', N'Equity', N'Booking Holdings Inc')
INSERT [dbo].[SecurityMaster] ([SecID], [Ticker], [Sector], [Country], [SecurityType], [SecurityName]) VALUES (46, N'BKR', N'', N'US', N'Equity', N'Baker Hughes Co')
INSERT [dbo].[SecurityMaster] ([SecID], [Ticker], [Sector], [Country], [SecurityType], [SecurityName]) VALUES (47, N'CDNS', N'', N'US', N'Equity', N'Cadence Design Systems Inc')
INSERT [dbo].[SecurityMaster] ([SecID], [Ticker], [Sector], [Country], [SecurityType], [SecurityName]) VALUES (48, N'CEG', N'', N'US', N'Equity', N'Constellation Energy Corp')
INSERT [dbo].[SecurityMaster] ([SecID], [Ticker], [Sector], [Country], [SecurityType], [SecurityName]) VALUES (49, N'CHTR', N'', N'US', N'Equity', N'Charter Communications Inc')
INSERT [dbo].[SecurityMaster] ([SecID], [Ticker], [Sector], [Country], [SecurityType], [SecurityName]) VALUES (50, N'CMCSA', N'', N'US', N'Equity', N'Comcast Corp')
INSERT [dbo].[SecurityMaster] ([SecID], [Ticker], [Sector], [Country], [SecurityType], [SecurityName]) VALUES (51, N'COST', N'', N'US', N'Equity', N'Costco Wholesale Corp')
INSERT [dbo].[SecurityMaster] ([SecID], [Ticker], [Sector], [Country], [SecurityType], [SecurityName]) VALUES (52, N'CPRT', N'', N'US', N'Equity', N'Copart Inc')
INSERT [dbo].[SecurityMaster] ([SecID], [Ticker], [Sector], [Country], [SecurityType], [SecurityName]) VALUES (53, N'CRWD', N'', N'US', N'Equity', N'Crowdstrike Holdings Inc')
INSERT [dbo].[SecurityMaster] ([SecID], [Ticker], [Sector], [Country], [SecurityType], [SecurityName]) VALUES (54, N'CSCO', N'', N'US', N'Equity', N'Cisco Systems Inc')
INSERT [dbo].[SecurityMaster] ([SecID], [Ticker], [Sector], [Country], [SecurityType], [SecurityName]) VALUES (55, N'CSGP', N'', N'US', N'Equity', N'CoStar Group Inc')
INSERT [dbo].[SecurityMaster] ([SecID], [Ticker], [Sector], [Country], [SecurityType], [SecurityName]) VALUES (56, N'CSX', N'', N'US', N'Equity', N'CSX Corp')
INSERT [dbo].[SecurityMaster] ([SecID], [Ticker], [Sector], [Country], [SecurityType], [SecurityName]) VALUES (57, N'CTAS', N'', N'US', N'Equity', N'Cintas Corp')
INSERT [dbo].[SecurityMaster] ([SecID], [Ticker], [Sector], [Country], [SecurityType], [SecurityName]) VALUES (58, N'CTSH', N'', N'US', N'Equity', N'Cognizant Technology Solutions Corp')
INSERT [dbo].[SecurityMaster] ([SecID], [Ticker], [Sector], [Country], [SecurityType], [SecurityName]) VALUES (59, N'DDOG', N'', N'US', N'Equity', N'Datadog Inc')
INSERT [dbo].[SecurityMaster] ([SecID], [Ticker], [Sector], [Country], [SecurityType], [SecurityName]) VALUES (60, N'DLTR', N'', N'US', N'Equity', N'Dollar Tree Inc')
INSERT [dbo].[SecurityMaster] ([SecID], [Ticker], [Sector], [Country], [SecurityType], [SecurityName]) VALUES (61, N'DXCM', N'', N'US', N'Equity', N'Dexcom Inc')
INSERT [dbo].[SecurityMaster] ([SecID], [Ticker], [Sector], [Country], [SecurityType], [SecurityName]) VALUES (62, N'EA', N'', N'US', N'Equity', N'Electronic Arts Inc')
INSERT [dbo].[SecurityMaster] ([SecID], [Ticker], [Sector], [Country], [SecurityType], [SecurityName]) VALUES (63, N'EBAY', N'', N'US', N'Equity', N'eBay Inc')
INSERT [dbo].[SecurityMaster] ([SecID], [Ticker], [Sector], [Country], [SecurityType], [SecurityName]) VALUES (64, N'ENPH', N'', N'US', N'Equity', N'Enphase Energy Inc')
INSERT [dbo].[SecurityMaster] ([SecID], [Ticker], [Sector], [Country], [SecurityType], [SecurityName]) VALUES (65, N'EXC', N'', N'US', N'Equity', N'Exelon Corp')
INSERT [dbo].[SecurityMaster] ([SecID], [Ticker], [Sector], [Country], [SecurityType], [SecurityName]) VALUES (66, N'FANG', N'', N'US', N'Equity', N'Diamondback Energy Inc')
INSERT [dbo].[SecurityMaster] ([SecID], [Ticker], [Sector], [Country], [SecurityType], [SecurityName]) VALUES (67, N'FAST', N'', N'US', N'Equity', N'Fastenal Co')
INSERT [dbo].[SecurityMaster] ([SecID], [Ticker], [Sector], [Country], [SecurityType], [SecurityName]) VALUES (68, N'FTNT', N'', N'US', N'Equity', N'Fortinet Inc')
INSERT [dbo].[SecurityMaster] ([SecID], [Ticker], [Sector], [Country], [SecurityType], [SecurityName]) VALUES (69, N'GEHC', N'', N'US', N'Equity', N'GE HealthCare Technologies Inc')
INSERT [dbo].[SecurityMaster] ([SecID], [Ticker], [Sector], [Country], [SecurityType], [SecurityName]) VALUES (70, N'GFS', N'', N'US', N'Equity', N'GLOBALFOUNDRIES Inc')
INSERT [dbo].[SecurityMaster] ([SecID], [Ticker], [Sector], [Country], [SecurityType], [SecurityName]) VALUES (71, N'GILD', N'', N'US', N'Equity', N'Gilead Sciences Inc')
INSERT [dbo].[SecurityMaster] ([SecID], [Ticker], [Sector], [Country], [SecurityType], [SecurityName]) VALUES (72, N'GOOG', N'', N'US', N'Equity', N'Alphabet Inc')
INSERT [dbo].[SecurityMaster] ([SecID], [Ticker], [Sector], [Country], [SecurityType], [SecurityName]) VALUES (73, N'HON', N'', N'US', N'Equity', N'Honeywell International Inc')
INSERT [dbo].[SecurityMaster] ([SecID], [Ticker], [Sector], [Country], [SecurityType], [SecurityName]) VALUES (74, N'IDXX', N'', N'US', N'Equity', N'IDEXX Laboratories Inc')
INSERT [dbo].[SecurityMaster] ([SecID], [Ticker], [Sector], [Country], [SecurityType], [SecurityName]) VALUES (75, N'ILMN', N'', N'US', N'Equity', N'Illumina Inc')
INSERT [dbo].[SecurityMaster] ([SecID], [Ticker], [Sector], [Country], [SecurityType], [SecurityName]) VALUES (76, N'INTC', N'', N'US', N'Equity', N'Intel Corp')
INSERT [dbo].[SecurityMaster] ([SecID], [Ticker], [Sector], [Country], [SecurityType], [SecurityName]) VALUES (77, N'INTU', N'', N'US', N'Equity', N'Intuit Inc')
INSERT [dbo].[SecurityMaster] ([SecID], [Ticker], [Sector], [Country], [SecurityType], [SecurityName]) VALUES (78, N'ISRG', N'', N'US', N'Equity', N'Intuitive Surgical Inc')
INSERT [dbo].[SecurityMaster] ([SecID], [Ticker], [Sector], [Country], [SecurityType], [SecurityName]) VALUES (79, N'JD', N'', N'US', N'Equity', N'JD.com Inc ADR')
INSERT [dbo].[SecurityMaster] ([SecID], [Ticker], [Sector], [Country], [SecurityType], [SecurityName]) VALUES (80, N'KDP', N'', N'US', N'Equity', N'Keurig Dr Pepper Inc')
INSERT [dbo].[SecurityMaster] ([SecID], [Ticker], [Sector], [Country], [SecurityType], [SecurityName]) VALUES (81, N'KHC', N'', N'US', N'Equity', N'Kraft Heinz Co/The')
INSERT [dbo].[SecurityMaster] ([SecID], [Ticker], [Sector], [Country], [SecurityType], [SecurityName]) VALUES (82, N'KLAC', N'', N'US', N'Equity', N'KLA Corp')
INSERT [dbo].[SecurityMaster] ([SecID], [Ticker], [Sector], [Country], [SecurityType], [SecurityName]) VALUES (83, N'LCID', N'', N'US', N'Equity', N'Lucid Group Inc')
INSERT [dbo].[SecurityMaster] ([SecID], [Ticker], [Sector], [Country], [SecurityType], [SecurityName]) VALUES (84, N'LULU', N'', N'US', N'Equity', N'Lululemon Athletica Inc')
INSERT [dbo].[SecurityMaster] ([SecID], [Ticker], [Sector], [Country], [SecurityType], [SecurityName]) VALUES (85, N'MAR', N'', N'US', N'Equity', N'Marriott International Inc/MD')
INSERT [dbo].[SecurityMaster] ([SecID], [Ticker], [Sector], [Country], [SecurityType], [SecurityName]) VALUES (86, N'MCHP', N'', N'US', N'Equity', N'Microchip Technology Inc')
INSERT [dbo].[SecurityMaster] ([SecID], [Ticker], [Sector], [Country], [SecurityType], [SecurityName]) VALUES (87, N'MDLZ', N'', N'US', N'Equity', N'Mondelez International Inc')
INSERT [dbo].[SecurityMaster] ([SecID], [Ticker], [Sector], [Country], [SecurityType], [SecurityName]) VALUES (88, N'MELI', N'', N'US', N'Equity', N'MercadoLibre Inc')
INSERT [dbo].[SecurityMaster] ([SecID], [Ticker], [Sector], [Country], [SecurityType], [SecurityName]) VALUES (89, N'MNST', N'', N'US', N'Equity', N'Monster Beverage Corp')
INSERT [dbo].[SecurityMaster] ([SecID], [Ticker], [Sector], [Country], [SecurityType], [SecurityName]) VALUES (90, N'MRNA', N'', N'US', N'Equity', N'Moderna Inc')
INSERT [dbo].[SecurityMaster] ([SecID], [Ticker], [Sector], [Country], [SecurityType], [SecurityName]) VALUES (91, N'MRVL', N'', N'US', N'Equity', N'Marvell Technology Inc')
INSERT [dbo].[SecurityMaster] ([SecID], [Ticker], [Sector], [Country], [SecurityType], [SecurityName]) VALUES (92, N'NFLX', N'', N'US', N'Equity', N'Netflix Inc')
INSERT [dbo].[SecurityMaster] ([SecID], [Ticker], [Sector], [Country], [SecurityType], [SecurityName]) VALUES (93, N'NXPI', N'', N'US', N'Equity', N'NXP Semiconductors NV')
INSERT [dbo].[SecurityMaster] ([SecID], [Ticker], [Sector], [Country], [SecurityType], [SecurityName]) VALUES (94, N'ODFL', N'', N'US', N'Equity', N'Old Dominion Freight Line Inc')
INSERT [dbo].[SecurityMaster] ([SecID], [Ticker], [Sector], [Country], [SecurityType], [SecurityName]) VALUES (95, N'ON', N'', N'US', N'Equity', N'ON Semiconductor Corp')
INSERT [dbo].[SecurityMaster] ([SecID], [Ticker], [Sector], [Country], [SecurityType], [SecurityName]) VALUES (96, N'ORLY', N'', N'US', N'Equity', N'O''Reilly Automotive Inc')
INSERT [dbo].[SecurityMaster] ([SecID], [Ticker], [Sector], [Country], [SecurityType], [SecurityName]) VALUES (97, N'PANW', N'', N'US', N'Equity', N'Palo Alto Networks Inc')
INSERT [dbo].[SecurityMaster] ([SecID], [Ticker], [Sector], [Country], [SecurityType], [SecurityName]) VALUES (98, N'PAYX', N'', N'US', N'Equity', N'Paychex Inc')
INSERT [dbo].[SecurityMaster] ([SecID], [Ticker], [Sector], [Country], [SecurityType], [SecurityName]) VALUES (99, N'PCAR', N'', N'US', N'Equity', N'PACCAR Inc')
GO
INSERT [dbo].[SecurityMaster] ([SecID], [Ticker], [Sector], [Country], [SecurityType], [SecurityName]) VALUES (100, N'PDD', N'', N'US', N'Equity', N'PDD Holdings Inc ADR')
INSERT [dbo].[SecurityMaster] ([SecID], [Ticker], [Sector], [Country], [SecurityType], [SecurityName]) VALUES (101, N'PEP', N'', N'US', N'Equity', N'PepsiCo Inc')
INSERT [dbo].[SecurityMaster] ([SecID], [Ticker], [Sector], [Country], [SecurityType], [SecurityName]) VALUES (102, N'PYPL', N'', N'US', N'Equity', N'PayPal Holdings Inc')
INSERT [dbo].[SecurityMaster] ([SecID], [Ticker], [Sector], [Country], [SecurityType], [SecurityName]) VALUES (103, N'QCOM', N'', N'US', N'Equity', N'QUALCOMM Inc')
INSERT [dbo].[SecurityMaster] ([SecID], [Ticker], [Sector], [Country], [SecurityType], [SecurityName]) VALUES (104, N'REGN', N'', N'US', N'Equity', N'Regeneron Pharmaceuticals Inc')
INSERT [dbo].[SecurityMaster] ([SecID], [Ticker], [Sector], [Country], [SecurityType], [SecurityName]) VALUES (105, N'ROST', N'', N'US', N'Equity', N'Ross Stores Inc')
INSERT [dbo].[SecurityMaster] ([SecID], [Ticker], [Sector], [Country], [SecurityType], [SecurityName]) VALUES (106, N'SBUX', N'', N'US', N'Equity', N'Starbucks Corp')
INSERT [dbo].[SecurityMaster] ([SecID], [Ticker], [Sector], [Country], [SecurityType], [SecurityName]) VALUES (107, N'SGEN', N'', N'US', N'Equity', N'Seagen Inc')
INSERT [dbo].[SecurityMaster] ([SecID], [Ticker], [Sector], [Country], [SecurityType], [SecurityName]) VALUES (108, N'SIRI', N'', N'US', N'Equity', N'Sirius XM Holdings Inc')
INSERT [dbo].[SecurityMaster] ([SecID], [Ticker], [Sector], [Country], [SecurityType], [SecurityName]) VALUES (109, N'TEAM', N'', N'US', N'Equity', N'Atlassian Corp')
INSERT [dbo].[SecurityMaster] ([SecID], [Ticker], [Sector], [Country], [SecurityType], [SecurityName]) VALUES (110, N'TMUS', N'', N'US', N'Equity', N'T-Mobile US Inc')
INSERT [dbo].[SecurityMaster] ([SecID], [Ticker], [Sector], [Country], [SecurityType], [SecurityName]) VALUES (111, N'TSLA', N'', N'US', N'Equity', N'Tesla Inc')
INSERT [dbo].[SecurityMaster] ([SecID], [Ticker], [Sector], [Country], [SecurityType], [SecurityName]) VALUES (112, N'TTD', N'', N'US', N'Equity', N'Trade Desk Inc/The')
INSERT [dbo].[SecurityMaster] ([SecID], [Ticker], [Sector], [Country], [SecurityType], [SecurityName]) VALUES (113, N'TXN', N'', N'US', N'Equity', N'Texas Instruments Inc')
INSERT [dbo].[SecurityMaster] ([SecID], [Ticker], [Sector], [Country], [SecurityType], [SecurityName]) VALUES (114, N'VRSK', N'', N'US', N'Equity', N'Verisk Analytics Inc')
INSERT [dbo].[SecurityMaster] ([SecID], [Ticker], [Sector], [Country], [SecurityType], [SecurityName]) VALUES (115, N'VRTX', N'', N'US', N'Equity', N'Vertex Pharmaceuticals Inc')
INSERT [dbo].[SecurityMaster] ([SecID], [Ticker], [Sector], [Country], [SecurityType], [SecurityName]) VALUES (116, N'WBA', N'', N'US', N'Equity', N'Walgreens Boots Alliance Inc')
INSERT [dbo].[SecurityMaster] ([SecID], [Ticker], [Sector], [Country], [SecurityType], [SecurityName]) VALUES (117, N'WBD', N'', N'US', N'Equity', N'Warner Bros Discovery Inc')
INSERT [dbo].[SecurityMaster] ([SecID], [Ticker], [Sector], [Country], [SecurityType], [SecurityName]) VALUES (118, N'WDAY', N'', N'US', N'Equity', N'Workday Inc')
INSERT [dbo].[SecurityMaster] ([SecID], [Ticker], [Sector], [Country], [SecurityType], [SecurityName]) VALUES (119, N'XEL', N'', N'US', N'Equity', N'Xcel Energy Inc')
INSERT [dbo].[SecurityMaster] ([SecID], [Ticker], [Sector], [Country], [SecurityType], [SecurityName]) VALUES (120, N'ZM', N'', N'US', N'Equity', N'Zoom Video Communications Inc')
INSERT [dbo].[SecurityMaster] ([SecID], [Ticker], [Sector], [Country], [SecurityType], [SecurityName]) VALUES (121, N'ZS', N'', N'US', N'Equity', N'Zscaler Inc')
INSERT [dbo].[SecurityMaster] ([SecID], [Ticker], [Sector], [Country], [SecurityType], [SecurityName]) VALUES (122, N'AAPL', NULL, N'US', N'Equity', N'Apple')
SET IDENTITY_INSERT [dbo].[SecurityMaster] OFF
GO
